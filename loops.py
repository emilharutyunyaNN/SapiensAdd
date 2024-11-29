# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union
import os, shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, precision_recall_fscore_support
from mmengine.evaluator import Evaluator
from mmengine.logging import print_log
from mmengine.registry import LOOPS
from .amp import autocast
from .base_loop import BaseLoop
from .utils import calc_dynamic_intervals
import socket
import torch.nn.functional as F


@LOOPS.register_module()
class EpochBasedTrainLoop(BaseLoop):
    """Loop for epoch-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_epochs: int,
            val_begin: int = 1,
            val_interval: int = 1,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)
        self._max_epochs = int(max_epochs)
        assert self._max_epochs == max_epochs, \
            f'`max_epochs` should be a integer number, but get {max_epochs}.'
        self._max_iters = self._max_epochs * len(self.dataloader)
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> torch.nn.Module:
        """Launch training."""
        self.runner.call_hook('before_train')

        while self._epoch < self._max_epochs and not self.stop_training:
            self.run_epoch()

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._epoch >= self.val_begin
                    and self._epoch % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train')
        return self.runner.model

    def run_epoch(self) -> None:
        """Iterate one epoch."""
        self.runner.call_hook('before_train_epoch')

        self.runner.model.train()

        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        self.runner.call_hook('after_train_epoch')
        self._epoch += 1

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.

        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self.epoch + 1))
        self.val_interval = self.dynamic_intervals[step - 1]


class _InfiniteDataloaderIterator:
    """An infinite dataloader iterator wrapper for IterBasedTrainLoop.

    It resets the dataloader to continue iterating when the iterator has
    iterated over all the data. However, this approach is not efficient, as the
    workers need to be restarted every time the dataloader is reset. It is
    recommended to use `mmengine.dataset.InfiniteSampler` to enable the
    dataloader to iterate infinitely.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self._dataloader = dataloader
        print("dataloader: ", self._dataloader.batch_size)
        self._iterator = iter(self._dataloader)
        self._epoch = 0

    def __iter__(self):
        return self

    def __next__(self) -> Union[Sequence[Sequence[dict]], Sequence[dict]]:
        try:
            data = next(self._iterator)
        except StopIteration:
            print_log(
                'Reach the end of the dataloader, it will be '
                'restarted and continue to iterate. It is '
                'recommended to use '
                '`mmengine.dataset.InfiniteSampler` to enable the '
                'dataloader to iterate infinitely.',
                logger='current',
                level=logging.WARNING)
            self._epoch += 1
            if hasattr(self._dataloader, 'sampler') and hasattr(
                    self._dataloader.sampler, 'set_epoch'):
                # In case the` _SingleProcessDataLoaderIter` has no sampler,
                # or data loader uses `SequentialSampler` in Pytorch.
                self._dataloader.sampler.set_epoch(self._epoch)

            elif hasattr(self._dataloader, 'batch_sampler') and hasattr(
                    self._dataloader.batch_sampler.sampler, 'set_epoch'):
                # In case the` _SingleProcessDataLoaderIter` has no batch
                # sampler. batch sampler in pytorch warps the sampler as its
                # attributes.
                self._dataloader.batch_sampler.sampler.set_epoch(self._epoch)
            time.sleep(2)  # Prevent possible deadlock during epoch transition
            self._iterator = iter(self._dataloader)
            data = next(self._iterator)
            
        return data


@LOOPS.register_module()
class IterBasedTrainLoop(BaseLoop):
    """Loop for iter-based training.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        max_iters (int): Total training iterations.
        val_begin (int): The iteration that begins validating.
            Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1000.
        dynamic_intervals (List[Tuple[int, int]], optional): The
            first element in the tuple is a milestone and the second
            element is a interval. The interval is used after the
            corresponding milestone. Defaults to None.
    """

    def __init__(
            self,
            runner,
            dataloader: Union[DataLoader, Dict],
            max_iters: int,
            val_begin: int = 1,
            val_interval: int = 1000,
            dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        super().__init__(runner, dataloader)
        self._max_iters = int(max_iters)
        assert self._max_iters == max_iters, \
            f'`max_iters` should be a integer number, but get {max_iters}'
        self._max_epochs = 1  # for compatibility with EpochBasedTrainLoop
        self._epoch = 0
        self._iter = 0
        self.val_begin = val_begin
        self.val_interval = val_interval
        # This attribute will be updated by `EarlyStoppingHook`
        # when it is enabled.
        self.stop_training = False
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in visualizer will be '
                'None.',
                logger='current',
                level=logging.WARNING)
        # get the iterator of the dataloader
        self.dataloader_iterator = _InfiniteDataloaderIterator(self.dataloader)

        self.dynamic_milestones, self.dynamic_intervals = \
            calc_dynamic_intervals(
                self.val_interval, dynamic_intervals)

    @property
    def max_epochs(self):
        """int: Total epochs to train model."""
        return self._max_epochs

    @property
    def max_iters(self):
        """int: Total iterations to train model."""
        return self._max_iters

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def iter(self):
        """int: Current iteration."""
        return self._iter

    def run(self) -> None:
        """Launch training."""
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        while self._iter < self._max_iters and not self.stop_training:
            self.runner.model.train()

            data_batch = next(self.dataloader_iterator)
           # print("Data batch: ", data_batch)
            self.run_iter(data_batch)

            self._decide_current_val_interval()
            if (self.runner.val_loop is not None
                    and self._iter >= self.val_begin
                    and self._iter % self.val_interval == 0):
                self.runner.val_loop.run()

        self.runner.call_hook('after_train_epoch')
        self.runner.call_hook('after_train')
        return self.runner.model

    def run_iter(self, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=self._iter, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        outputs = self.runner.model.train_step(
            data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=self._iter,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1

    def _decide_current_val_interval(self) -> None:
        """Dynamically modify the ``val_interval``."""
        step = bisect.bisect(self.dynamic_milestones, (self._iter + 1))
        self.val_interval = self.dynamic_intervals[step - 1]


@LOOPS.register_module()
class ValLoop(BaseLoop):
    """Loop for validation.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 validation. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False) -> None:
        super().__init__(runner, dataloader)

        if isinstance(evaluator, (dict, list)):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            assert isinstance(evaluator, Evaluator), (
                'evaluator must be one of dict, list or Evaluator instance, '
                f'but got {type(evaluator)}.')
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16

    def run(self) -> dict:
        """Launch validation."""
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data
                from dataloader.
        """
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        # outputs should be sequence of BaseDataElement
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.val_step(data_batch)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class TestLoop(BaseLoop):
    """Loop for test.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or dict): A dataloader object or a dict to
            build a dataloader.
        evaluator (Evaluator or dict or list): Used for computing metrics.
        fp16 (bool): Whether to enable fp16 testing. Defaults to
            False.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 evaluator: Union[Evaluator, Dict, List],
                 fp16: bool = False):
        super().__init__(runner, dataloader)
        #print("dataset length : ", len(self.dataloader.dataset), self.dataloader.dataset[0])
        if isinstance(evaluator, dict) or isinstance(evaluator, list):
            self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
        else:
            self.evaluator = evaluator  # type: ignore
        if hasattr(self.dataloader.dataset, 'metainfo'):
            self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
            self.runner.visualizer.dataset_meta = \
                self.dataloader.dataset.metainfo
        else:
            print_log(
                f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                'metainfo. ``dataset_meta`` in evaluator, metric and '
                'visualizer will be None.',
                logger='current',
                level=logging.WARNING)
        self.fp16 = fp16

    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()
        #print("dataloader: ", self.dataloader.dataset[0])
        for idx, data_batch in enumerate(self.dataloader):
            #print("Data batch: ", len(data_batch), "---", len(data_batch[0]))
            self.run_iter(idx, data_batch)



        # compute metrics
        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        # predictions should be sequence of BaseDataElement
        #print(self.runner.temporal_model, type(self.runner.temporal_model),"----")
        #return
        with autocast(enabled=self.fp16):
            outputs = self.runner.model.test_step(data_batch)
            #print("Outputs: ", outputs)
            
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
     
@LOOPS.register_module()   
class TrainLSTMLoop(IterBasedTrainLoop):
    def __init__(self,
                 runner,
                 dataloader: Union[DataLoader, Dict],
                 max_iters: int= 500,
                 fp16: bool = False,
                val_begin: int = 1,
                val_interval: int = 25,
                dynamic_intervals: Optional[List[Tuple[int, int]]] = None,
                log_file = "/home/emil/Keypoints/Sapiens/sapiens/pose/scripts/test/coco/sapiens_1b/lstm_train/trainlog_all21_b16.log",
                grad_log = "/home/emil/Keypoints/Sapiens/sapiens/pose/scripts/test/coco/sapiens_1b/lstm_train/trainlog_all21_b16_grads.log") -> None:
                
        super().__init__(runner, dataloader, max_iters,val_begin,val_interval, dynamic_intervals)
        
        if os.path.exists(log_file):
            os.remove(log_file)
        if os.path.exists(grad_log):
            os.remove(grad_log)
        self.fp16 = fp16
        print("mixed precision: ", self.fp16)
        self.lstm_optimizer = torch.optim.Adam(self.runner.temporal_model.parameters(), lr=1e-4)
        self.val_dataloader = self.runner._val_loop.dataloader
        print("val dataloader: ", self.val_dataloader)
        self.temporal_device = self.runner.cfg.get('device_map')['temporal']
        self.loss_fn = torch.nn.CrossEntropyLoss(weight= torch.tensor([0.5701219512195123, 0.6678571428571429, 0.5974440894568691, 0.6515679442508712, 0.5582089552238807, 0.8862559241706162, 1.0, 0.7056603773584906, 0.6254180602006689, 0.42596810933940776, 0.6111111111111112, 0.7923728813559323, 0.9739583333333334, 0.5033647375504711, 0.6404109589041096, 0.7991452991452992, 0.9257425742574258, 0.5137362637362638, 0.5718654434250765, 0.6726618705035972, 0.6849816849816851], device=self.temporal_device)) # 
        self.logger = logging.getLogger('TrainLSTMLoop')
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        self.grad_logger = logging.getLogger('TrainLSTMLoopGrads')
        ghandler = logging.FileHandler(grad_log)
        gformatter = logging.Formatter('%(asctime)s - %(message)s')
        ghandler.setFormatter(gformatter)
        self.grad_logger.addHandler(ghandler)
        self.grad_logger.setLevel(logging.INFO)
    def run(self) -> dict:
        self.runner.call_hook('before_train')
        # In iteration-based training loop, we treat the whole training process
        # as a big epoch and execute the corresponding hook.
        self.runner.call_hook('before_train_epoch')
        print("initial model: ",self.runner.model)
        self.runner.model.eval()
        self.runner.temporal_model.train()
        #idx = 0
        torch.autograd.set_detect_anomaly(True)
        for name, param in self.runner.temporal_model.named_parameters():
            #print("---", name, ">>>", param.requires_grad)
            if param.grad is not None:
                grad_mean = param.grad.abs().mean().item()
                #print(name, "---", grad_mean)
                self.grad_logger.info(f"Pramater {name}: {grad_mean:.4f}")
            else:
                self.grad_logger.info(f"Pramater {name} has no Gradient")
        
        while self._iter < self._max_iters and not self.stop_training:
            #and self._iter != 0:
            
            #check_mem()
            
            
            data_batch = next(self.dataloader_iterator)
            
            #print(data_batch)
            #eturn 
            #print([dt["data_samples"].img_id for dt in data_batch[0]])
            #print([dt["data_samples"].img_id for dt in data_batch[1]])
            #return
            #check_mem()
            batch_size = len(data_batch)
            
            #print("databatch: ",self.dataloader, len(data_batch), data_batch[0])
            #return
            #continue
            #print("len data batch: ", len(data_batch[0]), len(data_batch))
            with torch.no_grad():
                with autocast(enabled=False):
                    pose_outputs = self.runner.model.test_step(data_batch)
            
            #check_mem()    
            #print("---", len(pose_outputs), len(pose_outputs[0]), len(pose_outputs[0][0]))
            #print("pose:", type(pose_outputs[0][0].pred_fields.heatmaps),pose_outputs[0][0].pred_fields.heatmaps.shape)
            #print(torch.sigmoid(pose_outputs[0][0].pred_fields.heatmaps).min().item(), torch.sigmoid(pose_outputs[0][0].pred_fields.heatmaps).max().item())
            #return
            # run lstm on the sequence of heatmaps
            #pose_outputs = pose_outputs.to()
            if batch_size>1:
                print("bs")
                lstm_input = [pose_output[0] for pose_output in pose_outputs]
                target = torch.stack([pose_output[1][0] for pose_output in pose_outputs], dim = 0)
            else:
                lstm_input = pose_outputs[0]
                target = pose_outputs[1][0]
           # check_mem()
            target = target.to(f"cuda:{self.temporal_device}")
            #check_mem()
            #print("pose output: ", pose_outputs[0][0], type(pose_outputs[0]))
            #print("lstm_input: ", len(lstm_input))
            #print("lstm input: ", min(lstm_input[0]))
            lstm_in, lstm_vec_input = self.process_x(lstm_input)
            #return
            #check_mem()
            #print(type(type(self.runner.temporal_model)), self.runner.temporal_model)
            if "PoseVecLSTM" in str(type(self.runner.temporal_model)):
                lstm_input = lstm_vec_input
                #print("lstm vec shape: ", lstm_input.shape)
            else:
                lstm_input = lstm_in
            #print("lstm input: ", lstm_input[0].min().item())
            lstm_input = lstm_input.to(f"cuda:{self.temporal_device}")
            #self.runner.temporal_model = self.runner.temporal_model.to(self.temporal_device)
            #check_mem()
            lstm_out = self.runner.temporal_model(lstm_input)
            #print("lstm out: ", lstm_out, lstm_out.shape, target.shape)
            # take the class of the sequence as target to predict
            #check_mem()
            
            #print("LSTM out:" , lstm_out, lstm_out.device)
            #print("target: ", target.shape, target.device)
            #target = target.to(device = lstm_out.device)
            #print(len())# Define target based on application
            #print(target.device, lstm_out.device)
            if len(target.shape) >1:
                target = target.view(-1)
            print(lstm_out.shape, target.shape)
            loss = self.loss_fn(lstm_out, target)
            #del target, pose_outputs
            
            
            # Backpropagate and optimize
            self.lstm_optimizer.zero_grad()
            loss.backward()
            for name, param in self.runner.temporal_model.named_parameters():
                if param.grad is not None:
                    print(f"Gradient norm of {name}: {param.grad.norm().item()}")
            self.lstm_optimizer.step()
            
            for name, param in self.runner.temporal_model.named_parameters():
                if param.grad is not None:
                    grad_mean = param.grad.abs().mean().item()
                    print(f" {name} grad : ", grad_mean)
                    self.grad_logger.info(f"Pramater {name}: {grad_mean:.4f}")
                else:
                    self.grad_logger.info(f"Pramater {name} has no Gradient")
            torch.cuda.empty_cache()
            #print("LSTM loss:", loss.item())
            #print("True class and the predicted: ", target, " ---- ", torch.argmax(lstm_out, dim=1) )
            
            #self.runner.call_hook(
             #   'after_train_iter',
             #   batch_idx=self._iter,
             #   data_batch=data_batch,
              #  outputs={"loss": loss.item(),
                #         "max_class_index": torch.argmax(lstm_out, dim=1).item()
                ##         })
            loss_value = loss.item()
            self.logger.info(f"Iteration {self._iter}, Loss: {loss_value}")
            #print(f"LSTM loss: {loss_value}")
            self._iter += 1
            
            if self._iter % self.val_interval == 0 and self._iter != 0:
                self.validate()
            
            
            
        #self.runner.call_hook('after_train_epoch')
        #self.runner.call_hook('after_train')
        print("-------------- DONE------------------")
        return self.runner.temporal_model
    def process_pose_outputs(self, pose_outputs):
        # Process pose outputs to be compatible with LSTM input
        # Example: Flatten or rearrange as necessary
        return torch.flatten(pose_outputs, start_dim=2)  # Adjust as required
    

    def process_x(self, x):
        
        if isinstance(x, list) and not isinstance(x[0], list):
            x = [predinst.pred_fields.heatmaps for predinst in x]
            x = torch.stack(x, dim=0)
        elif isinstance(x, list) and isinstance(x[0], list): 
            #print("here list")
            x_new = []
            x_vec_new = []
            for sample in x:
                sample_kp_sc = [predinst.pred_instances.keypoint_scores for predinst in sample]
                sample_kp = [predinst.pred_instances.keypoints[0] for predinst in sample]
                sample_img_id = [predinst.img_path for predinst in sample]
                #print(sample[0])
                #print(sample_kp)
                lstm_kp_data = np.stack(sample_kp, axis = -2)
                lsmt_data = torch.tensor(lstm_kp_data, dtype=torch.float32)
                lsmt_data = lsmt_data.view(1,4,-1)
                
                #print(lstm_data, lstm_data.shape)
                #print("smpl: ", list(zip(sample_kp,sample_img_id)))
                #return
                #return 
                sample = [torch.sigmoid(predinst.pred_fields.heatmaps) for predinst in sample]
                #print
                #print(len(sample), sample[0].shape)
                #min_vls = sample[0].view(sample[0].size(0), -1).min(dim=1, keepdim = True)
        #print("means: ", mean_vls.shape, mean_vls)
                #max_vls = sample[0].view(sample[0].size(0), -1).max(dim=1, keepdim = True)
                #print("inp: ", len(inp))
                
                #print("min: ", sample[0].min().item(), "\n", "max values: ", sample[0].max().item())
                sample = [self.maxpool_heatmaps(smpl) for smpl in sample]
                #print("sample: ",sample[0].shape)
                #if len(sample[0].shape)<3:
                  #  sample = [spl.unsqueeze(0) for spl in sample]
                #sample = [F.max_pool2d(heatmap, kernel_size=4, stride=4) for heatmap in sample]
                #sample = [spl.squeeze(0) for spl in sample]
                #print("sample: ",sample[0].shape)
                #sample = [self.normalize(torch.sigmoid(one)) for one in sample]
                #return
                sample = torch.stack(sample, dim = 0)
                #print("sample: ",sample.shape)   
                x_new.append(sample)
                x_vec_new.append(lsmt_data)
            x_maps = torch.stack(x_new, dim=0)
            x_vecs = torch.stack(x_vec_new, dim=0)
            #print(x_new)
            if x_vecs.dim()>=4:
                x_vecs = x_vecs.squeeze(1)
            
        # print("x: ", x_maps.shape, x_vecs.shape)
        
        return x_maps, x_vecs
    
    def validate(self):
        
        print("-------- VALIDATION STARTED --------")
        self.runner.temporal_model.eval()  # Switch to evaluation mode
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            print(len(self.val_dataloader))
            for val_batch in self.val_dataloader:  # Loop through validation data
                # Process inputs and targets
                #print("val batch: ", val_batch)
                
                pose_outputs = self.runner.model.test_step(val_batch)
                
                if len(val_batch) >1:
                    lstm_input = [pose_output[0] for pose_output in pose_outputs]
                    target = torch.stack([pose_output[1][0] for pose_output in pose_outputs], dim = 0)
                else:
                    lstm_input = pose_outputs[0]
                    target = pose_outputs[1][0]    
                    
                lstm_in, lstm_vec_input = self.process_x(lstm_input)
                
                if "PoseVecLSTM" in str(type(self.runner.temporal_model)):
                    lstm_input = lstm_vec_input
                    #print("lstm vec shape: ", lstm_input.shape)
                else:
                    lstm_input = lstm_in
                lstm_input = lstm_input.to(f"cuda:{self.temporal_device}")
                target = target.to(f"cuda:{self.temporal_device}")

                # Forward pass
                lstm_out = self.runner.temporal_model(lstm_input)
                
                if len(target.shape) >1:
                    target = target.view(-1)
                loss = self.loss_fn(lstm_out, target)
                val_loss += loss.item()

                # Store predictions and targets
                preds = torch.argmax(lstm_out, dim=1)
                print("val pred: ",preds, "---", target )

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        # Compute average loss
        avg_val_loss = val_loss / len(self.val_dataloader)

        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_targets, all_preds, average='weighted', zero_division=0
        )
        report = classification_report(
            all_targets, all_preds, target_names=['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9', 'class_10', 'class_11', 'class_12', 'class_13', 'class_14', 'class_15', 'class_16', 'class_17', 'class_18', 'class_19', 'class_20']
        )
        correct_predictions = sum(p == t for p, t in zip(all_preds, all_targets))
        total_predictions = len(all_targets)
        accuracy = correct_predictions / total_predictions
        # Log results
        self.logger.info(f"Validation - Loss: {avg_val_loss:.4f}")
        self.logger.info(f"Weighted Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"\n{report}")
        print(f"Validation - Loss: {avg_val_loss:.4f}")
        print(f"Weighted Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(report)
        
        

        self.runner.temporal_model.train()  # Switch back to training mode
            #return torch.stack(target_lst, dim=0)
            
    def normalize(self, data):
        #print("input: ", data.shape)
        #print(d)
        mean_vls = data.view(data.size(0), -1).mean(dim=1, keepdim = True)
        #print("means: ", mean_vls.shape, mean_vls)
        std_vls = data.view(data.size(0), -1).std(dim=1, keepdim = True)
        #print("stds:", std_vls.shape, std_vls)
        mean_vals = mean_vls.view(-1,1,1)
        std_vals = std_vls.view(-1,1,1)
       # print("means view: ", mean_vals.shape, mean_vals)
        #print("stds view: ", mean_vals.shape)
        epsilon = 1e-8
        data_normalized = (data-mean_vals)/(std_vals+ epsilon)
        return data_normalized
    
    def maxpool_heatmaps(self, heatmaps):
        """
        Given a list of 17 heatmaps, perform max pooling to aggregate them into a single heatmap.
        
        Args:
            heatmaps_list (list of torch.Tensor): A list of heatmaps, each of shape (C, H, W).
            
        Returns:
            torch.Tensor: The aggregated heatmap of shape (C, H, W).
        """
        # Stack all heatmaps into a single tensor of shape (17, C, H, W)
        #stacked_heatmaps = torch.stack(heatmaps_list, dim=0)  # Shape: (17, C, H, W)
        
        # Perform max pooling across the 17 heatmaps (dim=0), element-wise max pooling
        pooled_heatmap, _ = torch.max(heatmaps, dim=0)  # Shape: (C, H, W)
        
        return pooled_heatmap
            
        
def check_mem():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # Iterate through GPUs and check memory
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Total memory: {torch.cuda.get_device_properties(i).total_memory / (1024 ** 3):.2f} GB")
        print(f"  Allocated memory: {torch.cuda.memory_allocated(i) / (1024 ** 3):.2f} GB")
        print(f"  Cached memory: {torch.cuda.memory_reserved(i) / (1024 ** 3):.2f} GB")
