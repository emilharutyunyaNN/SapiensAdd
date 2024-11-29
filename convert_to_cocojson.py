import json 
from pathlib import Path
import numpy as np 

# a code online, pending
"""
aedet_json='/home/x/Workspace/pose-ae-train/exp/och_val_norefine_ms/dt.json'
ochuman_json='/home/x/Data/OCHuman/ochuman_coco_format_val_range_0.00_1.00.json'

# SCORE_THRESH=0.2
# SCORE_THRESH=0.3
SCORE_THRESH=0.4
# SCORE_THRESH=0.5
# SCORE_THRESH=0.6
# SCORE_THRESH=0.7
# SCORE_THRESH=0.8
# SCORE_THRESH=0.9

och_path = Path(ochuman_json)
det_path = Path(aedet_json)

out_path = och_path.parent / f'{och_path.stem}-{det_path.parent.stem}-score{SCORE_THRESH}.json'

with det_path.open('r') as rf:
    det_list = json.load(rf)

with och_path.open('r') as rf: 
    och_dict = json.load(rf)

img_ids = [img['id'] for img in och_dict['images']]

new_annots = []
for img_id in img_ids:
    new_dets = False 
    for det in det_list: 
        if det['image_id'] != img_id: 
            continue
        if det['score'] < SCORE_THRESH:
            continue
        kpts = np.array(det['keypoints'])
        assert np.equal(kpts[2::3],1).all()
        kpts[2::3] = 2
        assert len(kpts) == 17*3

        annot = {'image_id': det['image_id'], 
                'area': None, 
                'num_keypoints': 0, 
                'iscrowd': 0, 
                'id': det['id'], 
                'category_id': 1, 
                'keypoints': kpts.tolist(), 
                'segmentation': [[]], 
                'bbox': []
                }

        assert annot['image_id'] in img_ids
        new_annots.append(annot)
        new_dets = True
    if not new_dets:
        empty = {'image_id': img_id, 
                'area': None, 
                'num_keypoints': 0, 
                'iscrowd': 0, 
                'id': None, 
                'category_id': 1, 
                'keypoints': [0.]*(17*3), 
                'segmentation': [[]], 
                'bbox': []
                }
        new_annots.append(empty)

och_dict['annotations'] = new_annots
with out_path.open('w') as wf: 
    json.dump(och_dict, wf)

print('Written to', out_path)
"""

# Here I am going to make a json bbox detector annotation to use in the OCHuman evaluation
import os
def show_content(file):
    file = os.path.abspath(file)
    with open(file, 'r') as f:
        data = json.load(f)
    if type(data) is list:
        return data[0].keys() 
    
    #if "annotations" in data.keys():
       # bbox_list = [ann["bbox"] for ann in data['annotations']]
        #return data.keys(), data['annotations'][0].keys() , len(bbox_list)

    #if "Crowdpose" in file:
       # return data.keys(), data['images']
    if data['annotations']:
 #       print("--")
        return data.keys(), [dt['file_name'] for dt in data['images']]
    return data.keys() 

import shutil
def make_validation(dir, img_list):
    abs_dir = os.path.abspath(dir)
    extract_list = os.listdir(abs_dir)
    val_imgs = []
    for path in extract_list:
        if path in img_list:
            val_imgs.append(path)
            
    val_dirp = os.path.dirname(dir)
    val_dir = os.path.join(val_dirp, "val")
    os.makedirs(val_dir, exist_ok=True)
    for path in val_imgs:
        src = os.path.join(abs_dir,path)
        dst = os.path.join(val_dir, path)
        print("-----", src, dst)
        shutil.move(src, dst)  
    
#lst = show_content("/home/emil/Keypoints/Sapiens/Crowdpose/annotations/crowdpose_val.json")[1]
#print(lst)
#make_validation("/home/emil/Keypoints/Sapiens/Crowdpose/images", lst)

#print("----- bbox coco: ", show_content("/home/emil/Keypoints/Sapiens/Coco/person_detection_results/COCO_val2017_detections_AP_H_70_person_new.json"))
#print("----- annot coco: ", show_content("/home/emil/Keypoints/Sapiens/Coco/annotations/coco_wholebody_val_v1.0.json"))
#print("\n", "----- Annotations OCHuman -----", "\n")
#print("----- ochuman.json: ", show_content("/home/emil/Keypoints/Sapiens/OCHuman/person_detection_results/ochuman.json"), "\n")
#print("----- val ochuman: ", show_content("/home/emil/Keypoints/Sapiens/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json"))
#print("\n", "------ Annotations Crowdpose ------", "\n")
#print("----- val crowdpose: ", show_content("/home/emil/Keypoints/Sapiens/Crowdpose/annotations/crowdpose_val.json"))
def make_bbox_ann(och_val, path = "."):
    bbox_annots = []
    och_val = os.path.abspath(och_val)
    print(och_val)
    with open(och_val, 'r') as f:
        data = json.load(f)
    print(len(data['annotations']))
    for ann in data["annotations"]:
        if 'segmentations' in ann.keys():
            bbox_annots.append({
                "image_id": ann['image_id'],
                "category_id": [ann["bbox"][0], ann["bbox"][1], ann["bbox"][0]+ ann["bbox"][2], ann["bbox"][1]+ ann["bbox"][3]],
                "bbox": ann["bbox"],
                "score": 1.0,
                "class": ann["class"],
                "segmentation": ann["segmentation"]
            })
        else:
            if ann["image_id"] == 1:
                print(ann["image_id"])
            bbox_annots.append({
                "image_id": ann['image_id'],
                "category_id": ann['category_id'],
                "bbox": [ann["bbox"][0], ann["bbox"][1], ann["bbox"][0]+ ann["bbox"][2], ann["bbox"][1]+ ann["bbox"][3]],
                "score": 1.0,
                "class": ann["class"]
            })
            
    print(len(bbox_annots))
    if "person_detection_results" not in os.listdir(path):
        os.makedirs("person_detection_results")
    with open(f"{path}/bbox_all_val_new_final.json", "w") as f:
        json.dump(bbox_annots, f, indent=4)
        
        
#make_bbox_ann("/home/emil/Keypoints/Sapiens/OCHuman/annotations/ochuman_coco_format_val_range_0.00_1.00.json")

#make_bbox_ann("/home/emil/Keypoints/Sapiens/Crowdpose/annotations/crowdpose_val.json", "/home/emil/Keypoints/Sapiens/Crowdpose")

make_bbox_ann("./coco_all_val_new_final.json")

