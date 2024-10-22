# Sapiens model is a big model and it takes a lot of time to play around and see how it works
# and how accurate it is therefore one way of having a quicker inference and testing is artificially 
# creating smaller dataset chunks and corresponding annotations, provided they have been downloaded, if not 
# follow the following guide https://github.com/facebookresearch/sapiens/blob/main/docs/finetune/POSE_README.md
# In my case I am looking into pose estimation so I downloaded:
#       - COCO_WholeBody dataset and 133 checkpoints annotations
#       - Val2017 dataset with 5000 imgs
#       - BBox detector checkpoint

import os
import json
import random
import shutil
# Since it will take a day to run a testing with 3 GPU s 14816 iterations: 
# therefore I will randomly take N percentage of the images in validation set and their annotations
def new_set(val_path, prctg=1):
    val_full = os.path.abspath(val_path)
    curr_vals = os.listdir(val_full)
    cardinality = len(curr_vals)
    new_crd = int(prctg*cardinality/100)
    new_vals = random.sample(curr_vals, new_crd) 
    full_paths_new_vals_old = [os.path.join(val_full, val) for val in new_vals]
    new_val_dir = f"{val_full}_"
   
    if os.path.isdir(new_val_dir) and os.listdir(new_val_dir) != []:
        shutil.rmtree(new_val_dir)
    os.makedirs(new_val_dir, exist_ok=True)
            
    
    #full_paths_new_vals_new = [os.path.join(val_full+"_", val) for val in new_vals]
    for path_old in full_paths_new_vals_old:
        shutil.copy(path_old, new_val_dir)
    print(f"Copied {len(full_paths_new_vals_old)} images to {new_val_dir}.")
    print(len(os.listdir(new_val_dir)))
    return new_vals   
    
def save_json(data, filename):
    with open(filename, "w") as save_file:
        json.dump(data, save_file)
    print(f"Saved {filename}")    

def process_ann(file_ann,file_bbx, img_list = None):
    #new_list = new_set(val_path)
    with open(file_ann) as f1:
        data_ann = json.load(f1)
        
    with open(file_bbx) as f2:
        data_bbx = json.load(f2)
        
        
    filtered_annotations = {
        'info': data_ann['info'],  # Keep the 'info' part
        'licenses': data_ann['licenses'],  # Keep the 'licenses' part
        'images': [],
        'annotations': [],
        'categories': data_ann['categories']  # Keep the 'categories' part
    }
    img_ids_to_keep = set()
    for img in data_ann['images']:
        if img['file_name'] in img_list:
            filtered_annotations['images'].append(img)
            print("ID: ", img['id'])
            img_ids_to_keep.add(img['id'])
    filtered_bbox = [annotation for annotation in data_bbx if annotation['image_id'] in img_ids_to_keep]
    for ann in data_ann['annotations']:
        if ann['image_id'] in img_ids_to_keep:
            filtered_annotations['annotations'].append(ann)
    #print("List: ", img_list)
    #print("Filtered images:", filtered_annotations['images'])  # Check final images
    #print("Filtered annotations count:", len(filtered_annotations['annotations'])) 
    dir_to_save = os.path.dirname(file_ann)
    new_file = dir_to_save+"/"+ os.path.basename(file_ann)[:-5]+"_new.json"
    if os.path.basename(file_ann)[:-5]+"_new.json" in os.listdir(dir_to_save):
        os.remove(new_file)
    save_json(filtered_annotations, new_file)
    
    dir_to_save = os.path.dirname(file_bbx)
    new_file = dir_to_save+"/"+ os.path.basename(file_bbx)[:-5]+"_new.json"
    if os.path.basename(file_bbx)[:-5]+"_new.json" in os.listdir(dir_to_save):
        os.remove(new_file)
    save_json(filtered_bbox, new_file)
    
    
imgs_to_keep = new_set("/home/emil/Keypoints/Sapiens/Coco/val2017")
process_ann("/home/emil/Keypoints/Sapiens/Coco/annotations/coco_wholebody_val_v1.0.json","/home/emil/Keypoints/Sapiens/Coco/person_detection_results/COCO_val2017_detections_AP_H_70_person.json", imgs_to_keep)

#print(os.listdir("/home/emil/Keypoints/Sapiens/Coco/val2017_"))

#with open("/home/emil/Keypoints/Sapiens/Coco/person_detection_results/COCO_val2017_detections_AP_H_70_person.json") as f:
 #   data = json.load(f)
    
#print(data[0].keys())
#print()
