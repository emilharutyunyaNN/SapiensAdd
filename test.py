# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import typing
import mmengine
from mmengine.config import Config, DictAction
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.hooks.logger_hook import LoggerHook
import torch
#print(f"Using device: {torch.cuda.current_device()}") 

import os

# Get the local rank (which determines which GPU this process will use)
#slocal_rank = int(os.getenv("LOCAL_RANK", 0))

# Set the current GPU device based on the local rank
#torch.cuda.set_device(local_rank)
"""device_ids = [1,4,5]
local_rank = int(os.environ['LOCAL_RANK'])
device = torch.device(f'cuda:{device_ids[local_rank]}')
print("--- ", local_rank, device)
torch.cuda.set_device(device)
print(f"Using device: {device}")"""
import torch
print(f"Number of visible GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"Logical GPU {i} is mapped to physical GPU {torch.cuda.get_device_name(i)}")


#### CUSTOM HOOK class for interation evaluation

def parse_device_map(device_map):
    """Parse the device map string into a dictionary."""
    mapping = {}
    for pair in device_map.split(","):
        key, value = pair.split(":")
        mapping[key.strip()] = int(value.strip())
    return mapping


class IterationHook(Hook):
    def __init__(self, interval=1, output_file = None):
        self.interval = interval
        self.output_file = output_file
    def after_test_iter(self, runner, batch_idx: int, data_batch: dict | tuple | list | None = None, outputs: typing.Sequence | None = None) -> None:
        #if runner.iter is None or runner.data_loader is None:
          #  print("skipping...")
            # Skip hook execution if runner isn't fully initialized
           # return

        if self.every_n_inner_iters(batch_idx, self.interval):
            
            runner.logger.info(f'Evaluating at iteration {runner.iter + 1}')
            
            print("Time for evaluation")
            
            if outputs is not None:
                
                #print("-----", outputs[0].pred_instances.keypoints)
                #print("keypoints: ", keypoints)
                #return
                #print("---", type(outputs[0].metainfo['img_id']), str(outputs[0].metainfo['img_id']))
                #return
                #print("Ground_truth: ", outputs[0].gt_instances)
                if self.output_file is not None:
                    # Append the metrics to the output file
                    with open(self.output_file, 'a') as f:
                        # {[(output.pred_instances.keypoints, output.pred_instances.keypoint_scores, str(output.metainfo['img_id'])) for output in outputs]}
                        f.write(f"Iteration {batch_idx + 1}: {outputs}\n")
                    print(f"Metrics saved for iteration {batch_idx + 1} to {self.output_file}")
                else:
                    print(f"Iteration {batch_idx + 1}: {outputs}")
            else:
                print(f"No metrics found at iteration {batch_idx + 1}")

#some = LoggerHook()
#some.after_val_epoch
#commenting kjdaisajjshss sdsaa

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMPose test (and eval) model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir', help='the directory to save evaluation results')
    parser.add_argument('--out', help='the file to save metric results.')
    parser.add_argument(
        '--dump',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--device-map',
        help='In case of non distributed training/testing the mapping for models.')
    parser.add_argument(
        '--temp',
        help='Tells us if we should use lstm or not. ')
    parser.add_argument(
        '--show-dir',
        help='directory where the visualization images will be saved.')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to display the prediction results in a window.')
    parser.add_argument(
        '--interval',
        type=int,
        default=1,
        help='visualize per interval samples.')
    parser.add_argument(
        '--wait-time',
        type=float,
        default=1,
        help='display time of every window. (second)')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def merge_args(cfg, args):
    """Merge CLI arguments to config."""

    cfg.launcher = args.launcher
    cfg.load_from = args.checkpoint
    cfg.device_map = parse_device_map(args.device_map)
    cfg.temp = True if args.temp is not None else False
    # -------------------- work directory --------------------
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # -------------------- visualization --------------------
    if args.show or (args.show_dir is not None):
        assert 'visualization' in cfg.default_hooks, \
            'PoseVisualizationHook is not set in the ' \
            '`default_hooks` field of config. Please set ' \
            '`visualization=dict(type="PoseVisualizationHook")`'

        cfg.default_hooks.visualization.enable = True
        cfg.default_hooks.visualization.show = args.show
        if args.show:
            cfg.default_hooks.visualization.wait_time = args.wait_time
        cfg.default_hooks.visualization.out_dir = args.show_dir
        cfg.default_hooks.visualization.interval = args.interval
        
        print("---", cfg.default_hooks.visualization)

    # -------------------- Dump predictions --------------------
    if args.dump is not None:
        assert args.dump.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        dump_metric = dict(type='DumpResults', out_file_path=args.dump)
        if isinstance(cfg.test_evaluator, (list, tuple)):
            cfg.test_evaluator = [*cfg.test_evaluator, dump_metric]
        else:
            cfg.test_evaluator = [cfg.test_evaluator, dump_metric]

    # -------------------- Other arguments --------------------
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    return cfg


def main():
    args = parse_args()
    
    # load config
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args)
    #print("cfg: ", cfg)
    #return
    # build the runner from config
    
   # print("config: ", cfg, "-------", args)
    #return 
    runner = Runner.from_cfg(cfg)

    """if args.out:
        class SaveMetricHook(Hook):
            def after_test_epoch(self, _, metrics=None):
                print("After test epoch hook called.")  # Debug line
                if metrics is not None:
                    print(metrics)
                    mmengine.dump(metrics, args.out)
                    print(f"Metrics saved to {args.out}")  # Debug line
                else:
                    print("No metrics found.")  # Debug line
        runner.register_hook(SaveMetricHook(), 'LOWEST')"""
    output_file = args.out if args.out else None
    print("--- out: ", output_file)
    iteration_hook = IterationHook(interval=10, output_file=output_file)  # Set interval as needed
    runner.register_hook(iteration_hook, priority='NORMAL')

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
