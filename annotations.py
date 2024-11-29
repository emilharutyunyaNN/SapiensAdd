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
    
    
#imgs_to_keep = new_set("/home/emil/Keypoints/Sapiens/Coco/val2017")
#process_ann("/home/emil/Keypoints/Sapiens/Coco/annotations/coco_wholebody_val_v1.0.json","/home/emil/Keypoints/Sapiens/Coco/person_detection_results/COCO_val2017_detections_AP_H_70_person.json", imgs_to_keep)

#print(os.listdir("/home/emil/Keypoints/Sapiens/Coco/val2017_"))

#with open("/home/emil/Keypoints/Sapiens/Coco/person_detection_results/COCO_val2017_detections_AP_H_70_person.json") as f:
 #   data = json.load(f)
    
#print(data[0].keys())
#print()

def ann_train_test_split(ann):
    prctg = 0.8
    frame = 4
    with open(ann, "r") as f:
        data = json.load(f)
        
    train_ann = dict()
    train_ann["images"] = []
    train_ann["annotations"] = []
    train_ann["categories"] = data["categories"]
    val_ann = dict()
    val_ann["images"] = []
    val_ann["annotations"] = []
    val_ann["categories"] = data["categories"]
    
    data["images"] = sorted(data["images"], key=lambda x: x["id"]) 
    data["annotations"] = sorted(data["annotations"], key=lambda x: x["image_id"])
    
    class_0 = dict()
    class_1 = dict()
    class_2 = dict()
    class_3 = dict()
    class_4 = dict()
    class_5 = dict()
    class_0["images"] = [dt for dt in data["images"] if dt["class"] == 0]
    class_1["images"] = [dt for dt in data["images"] if dt["class"] == 1]
    class_2["images"] = [dt for dt in data["images"] if dt["class"] == 2]
    class_3["images"] = [dt for dt in data["images"] if dt["class"] == 3]
    class_4["images"] = [dt for dt in data["images"] if dt["class"] == 4]
    class_5["images"] = [dt for dt in data["images"] if dt["class"] == 5]
    
    class_0["annotations"] = [dt for dt in data["annotations"] if dt["class"] == 0]
    class_1["annotations"] = [dt for dt in data["annotations"] if dt["class"] == 1]
    class_2["annotations"] = [dt for dt in data["annotations"] if dt["class"] == 2]
    class_3["annotations"] = [dt for dt in data["annotations"] if dt["class"] == 3]
    class_4["annotations"] = [dt for dt in data["annotations"] if dt["class"] == 4]
    class_5["annotations"] = [dt for dt in data["annotations"] if dt["class"] == 5]
    
    class_0["images"] = sorted(class_0["images"], key=lambda x: x["id"]) 
    class_0["annotations"] = sorted(class_0["annotations"], key=lambda x: x["image_id"])
    class_1["images"] = sorted(class_1["images"], key=lambda x: x["id"]) 
    class_1["annotations"] = sorted(class_1["annotations"], key=lambda x: x["image_id"])
    class_2["images"] = sorted(class_2["images"], key=lambda x: x["id"]) 
    class_2["annotations"] = sorted(class_2["annotations"], key=lambda x: x["image_id"])
    class_3["images"] = sorted(class_3["images"], key=lambda x: x["id"]) 
    class_3["annotations"] = sorted(class_3["annotations"], key=lambda x: x["image_id"])
    class_4["images"] = sorted(class_4["images"], key=lambda x: x["id"]) 
    class_4["annotations"] = sorted(class_4["annotations"], key=lambda x: x["image_id"])
    class_5["images"] = sorted(class_5["images"], key=lambda x: x["id"]) 
    class_5["annotations"] = sorted(class_5["annotations"], key=lambda x: x["image_id"])

    len_0im = len(class_0["images"])
    len_0ann = len(class_0["annotations"])
    len_0im_train = int(prctg*len_0im)
    len_0ann_train = int(prctg*len_0ann)
    
    len_1im = len(class_1["images"])
    len_1ann = len(class_1["annotations"])
    len_1im_train = int(prctg*len_1im)
    len_1ann_train = int(prctg*len_1ann)
    
    len_2im = len(class_2["images"])
    len_2ann = len(class_2["annotations"])
    len_2im_train = int(prctg*len_2im)
    len_2ann_train = int(prctg*len_2ann)
    
    len_3im = len(class_3["images"])
    len_3ann = len(class_3["annotations"])
    len_3im_train = int(prctg*len_3im)
    len_3ann_train = int(prctg*len_3ann)
    
    len_4im = len(class_4["images"])
    len_4ann = len(class_4["annotations"])
    len_4im_train = int(prctg*len_4im)
    len_4ann_train = int(prctg*len_4ann)
    
    len_5im = len(class_5["images"])
    len_5ann = len(class_5["annotations"])
    len_5im_train = int(prctg*len_5im)
    len_5ann_train = int(prctg*len_5ann)
    
    print(len_0im_train,len_1im_train,len_2im_train,len_3im_train,len_4im_train,len_5im_train)
    print(len_0ann_train,len_1ann_train,len_2ann_train,len_3ann_train,len_4ann_train,len_5ann_train)
    
    for dt in class_0["images"][:len_0im_train]:
        train_ann["images"].append(dt)
    for dt in class_0["annotations"][:len_0im_train]:
        train_ann["annotations"].append(dt)
        
    for dt in class_1["images"][:len_1im_train]:
        train_ann["images"].append(dt)
    for dt in class_1["annotations"][:len_1im_train]:
        train_ann["annotations"].append(dt)
        
    for dt in class_2["images"][:len_2im_train]:
        train_ann["images"].append(dt)
    for dt in class_2["annotations"][:len_2im_train]:
        train_ann["annotations"].append(dt)
        
    for dt in class_3["images"][:len_3im_train]:
        train_ann["images"].append(dt)
    for dt in class_3["annotations"][:len_3im_train]:
        train_ann["annotations"].append(dt)
        
    for dt in class_4["images"][:len_4im_train]:
        train_ann["images"].append(dt)
    for dt in class_4["annotations"][:len_4im_train]:
        train_ann["annotations"].append(dt)

    for dt in class_5["images"][:len_5im_train]:
        train_ann["images"].append(dt)
    for dt in class_5["annotations"][:len_5im_train]:
        train_ann["annotations"].append(dt)
        
    for dt in class_0["images"][len_0im_train:]:
        val_ann["images"].append(dt)
    for dt in class_0["annotations"][len_0im_train:]:
        val_ann["annotations"].append(dt)
        
    for dt in class_1["images"][len_1im_train:]:
        val_ann["images"].append(dt)
    for dt in class_1["annotations"][len_1im_train:]:
        val_ann["annotations"].append(dt)
        
    for dt in class_2["images"][len_2im_train:]:
        val_ann["images"].append(dt)
    for dt in class_2["annotations"][len_2im_train:]:
        val_ann["annotations"].append(dt)
        
    for dt in class_3["images"][len_3im_train:]:
        val_ann["images"].append(dt)
    for dt in class_3["annotations"][len_3im_train:]:
        val_ann["annotations"].append(dt)
        
    for dt in class_4["images"][len_4im_train:]:
        val_ann["images"].append(dt)
    for dt in class_4["annotations"][len_4im_train:]:
        val_ann["annotations"].append(dt)

    for dt in class_5["images"][len_5im_train:]:
        val_ann["images"].append(dt)
    for dt in class_5["annotations"][len_5im_train:]:
        val_ann["annotations"].append(dt)
        
    with open("coco_vids_train.json", "w") as t:
        json.dump(train_ann, t, indent=4)
        
    with open("coco_vids_val.json", "w") as v:
        json.dump(val_ann, v, indent=4)
        
        
    
    

#ann_train_test_split("/home/emil/Keypoints/Sapiens/six_vids/annotations/coco_vids.json")

def process_ann(ann_file, train = True):
    ann_abs = os.path.abspath(ann_file)
    with open(ann_abs, "r") as f:
        data = json.load(f)
        
        
    new_annot = dict()
    new_annot["categories"] = data["categories"]
    new_annot["images"] = []
    new_annot["annotations"] = []
    for dt in data["images"]:
        if dt["class"] != 5 and dt["class"] !=1:
            new_annot["images"].append(dt)
            
    for dt in data["annotations"]:
        if dt["class"] != 5 and dt["class"] !=1:
            new_annot["annotations"].append(dt)
            
    if train:
        with open("balanced_coco_train.json", "w") as f:
            json.dump(new_annot, f)
            
    else:
        with open("balanced_coco_val.json", "w") as f:
            json.dump(new_annot, f)
            
#process_ann("/home/emil/Keypoints/Sapiens/six_vids/annotations/coco_vids_train.json")
#process_ann("/home/emil/Keypoints/Sapiens/six_vids/annotations/coco_vids_val.json", False)


def change_classes(ann):
    ann_abs = os.path.abspath(ann)
    with open(ann_abs, "r") as f:
        data = json.load(f)
        
        
    for dt in data["images"]:
        if dt["class"] == 2:
            dt["class"] =1
        elif dt["class"] == 3:
            dt["class"] =2
        if dt["class"] == 4:
            dt["class"] =3
            
    for dt in data["annotations"]:
        if dt["class"] == 2:
            dt["class"] =1
        elif dt["class"] == 3:
            dt["class"] =2
        if dt["class"] == 4:
            dt["class"] =3
            
            
    with open(ann, "w") as f:
        json.dump(data, f)
    
        
#change_classes("/home/emil/Keypoints/Sapiens/six_vids/annotations/coco_vids_train.json")
#change_classes("/home/emil/Keypoints/Sapiens/six_vids/annotations/coco_vids_val.json")
#change_classes("/home/emil/Keypoints/Sapiens/six_vids/annotations/balanced_coco_train.json")
#change_classes("/home/emil/Keypoints/Sapiens/six_vids/annotations/balanced_coco_val.json")

def put_together(ann1, ann2):
    
    with open(ann1, "r") as a1:
        d1 = json.load(a1)
        
    with open(ann2, "r") as a2:
        d2 = json.load(a2)
        
    new_annot = dict()
    new_annot["categories"] = d1["categories"]
    new_annot["images"] = []
    new_annot["annotations"] = []
    
    for im in d1["images"]:
        new_annot["images"].append(im)
    
    for im in d2["images"]:
        new_annot["images"].append(im)
    
    for ann in d1["annotations"]:
        new_annot["annotations"].append(ann)
    
    for ann in d2["annotations"]:
        new_annot["annotations"].append(ann)
        
    new_annot["images"] = sorted(new_annot["images"], key=lambda x : x["id"])
    new_annot["annotations"] = sorted(new_annot["annotations"], key = lambda x: x["image_id"])
    
    frame = 4
    new_train_annot = dict()
    new_train_annot["categories"] = new_annot["categories"]
    new_train_annot["images"] = []
    new_train_annot["annotations"] = []
    
    new_val_annot = dict()
    new_val_annot["categories"] = new_annot["categories"]
    new_val_annot["images"] = []
    new_val_annot["annotations"] = []
    
    def data(group, img = True):
        #print(int(len(group)/frame))
        new_data = []
        for i in range(0, int(len(group)/frame)):
            new_data.append(group[i::int(len(group)/frame)])
        
        data_len_train = int(len(new_data)*0.8)    
        print("--", data_len_train)
        #random.shuffle(new_data)
        if img:
            new_train_annot["images"].extend(new_data[:data_len_train])
            new_val_annot["images"].extend(new_data[data_len_train:])
        else:
            new_train_annot["annotations"].extend(new_data[:data_len_train])
            new_val_annot["annotations"].extend(new_data[data_len_train:])
        print(len(new_train_annot["images"]), "----", len(new_val_annot["images"]))
        return new_data    
    
    """annots_trial_ims = new_annot["images"][:40]
    annots_trial_anns = new_annot["annotations"][:40]
    
    print("ims ----", data(annots_trial_ims))
    
    print("anns ----", data(annots_trial_anns))"""
    
    for i in range(0, len(new_annot["images"]), 40):
        current_group = new_annot["images"][i:i+40]
        data(current_group)
        
    for i in range(0, len(new_annot["annotations"]), 40):
        current_group = new_annot["annotations"][i:i+40]
        data(current_group, False)
    
    list_img_val = []
    list_ann_val = []    
    list_img_train = []
    list_ann_train = []
    for img in new_val_annot["images"]:
        for im in img:
            list_img_val.append(im)
            
    for img1 in new_train_annot["images"]:
        for im1 in img1:
            list_img_train.append(im1)
            
    for ann in new_val_annot["annotations"]:
        for an in ann:
            list_ann_val.append(an)
    
    
    for ann_a in new_train_annot["annotations"]:
        for aa in ann_a:
            list_ann_train.append(aa)
      
    new_train_annot["images"] = list_img_train
    new_train_annot["annotations"] = list_ann_train
    new_val_annot["images"] = list_img_val
    new_val_annot["annotations"] = list_ann_val
    with open("balanced_coco_train_new.json", "w") as f:
        json.dump(new_train_annot, f)
        
    with open("balanced_coco_val_new.json", "w") as g:
        json.dump(new_val_annot, g)
    
    
    #print(new_train_annot["images"][:10])
    #print(new_val_annot["images"][:10])
    #print(len(new_train_annot["images"]), "----", len(new_val_annot["images"]), "-----", len(new_annot["images"]))
    
#put_together("/home/emil/Keypoints/Sapiens/six_vids/annotations/balanced_coco_train.json", "/home/emil/Keypoints/Sapiens/six_vids/annotations/balanced_coco_val.json")
        
    
        
    
    
def load(ann):
    with open(ann, "r") as f:
        data = json.load(f)
    #print(data)
    print(len(data))
    try:
        print(len(data["images"]))
    except:
        return    
    
##load("/home/emil/Keypoints/Sapiens/six_vids/person_detection_results/bbox_annotations_val_new.json")
#load("/home/emil/Keypoints/Sapiens/six_vids/person_detection_results/bbox_annotations_train_new.json")
#load("/home/emil/Keypoints/Sapiens/six_vids/annotations/balanced_coco_val_new.json")


def check_dataset(dir):
    list_acts = os.listdir(dir)
    act_dict = dict()
    for a in list_acts:
        act_dict[a] = 0
        
    list_acts_plus = [os.path.join(dir, a) for a in list_acts if "." not in a]
    
    for a in list_acts_plus:
        print("action: ", a)
        list_emp_a = []
        for act in os.listdir(a):
            if act!= ".DS_Store" and act !=".AppleDouble":
                list_emp_a.append(os.path.join(a, act))
        
        print(len(list_emp_a), "----", [len(os.listdir(e))-1 for e in list_emp_a])
        #print("acts of a: ", acts_a)
        #print(a, "--- ", len(acts_a), [len(os.listdir(aa))-1 for aa in acts_a])
        #
        
#check_dataset("/home/emil/Keypoints/Sapiens/video_data/Rename_Images")   

from collections import defaultdict
def train_test_all(ann):
    
    with open(ann, "r") as f:
        data =json.load(f)
        
    data["images"] = sorted(data["images"], key = lambda x : x["id"])
    data["annotations"] = sorted(data["annotations"], key = lambda x: x["image_id"])
    
    frame = 4
    new_train_annot = dict()
    new_train_annot["categories"] = data["categories"]
    new_train_annot["images"] = []
    new_train_annot["annotations"] = []
    
    new_val_annot = dict()
    new_val_annot["categories"] = data["categories"]
    new_val_annot["images"] = []
    new_val_annot["annotations"] = []
    
    
    grouped_data = defaultdict(lambda: {"images": [], "annotations": []})

# Group images by file prefix (e.g., directory name)
    for image in data["images"]:
        file_group = os.path.dirname(image["file_name"])  # Extract directory as group key
        grouped_data[file_group]["images"].append(image)

    # Add annotations to the corresponding groups
    for annotation in data["annotations"]:
        # Find the image this annotation belongs to (via image_id)
        image = next((img for img in data["images"] if img["id"] == annotation["image_id"]), None)
        if image:
            file_group = os.path.dirname(image["file_name"])  # Use the image's directory to group
            grouped_data[file_group]["annotations"].append(annotation)

    # Convert defaultdict to a regular dictionary (optional)
    grouped_data = dict(grouped_data)
    
    
    def data_p(group, img = True):
        #print(int(len(group)/frame))
        new_data = []
        for i in range(0, int(len(group)/frame)):
            new_data.append(group[i::int(len(group)/frame)])
        
        data_len_train = int(len(new_data)*0.8)    
        print("--", data_len_train)
        #random.shuffle(new_data)
        if img:
            new_train_annot["images"].extend(new_data[:data_len_train])
            new_val_annot["images"].extend(new_data[data_len_train:])
        else:
            new_train_annot["annotations"].extend(new_data[:data_len_train])
            new_val_annot["annotations"].extend(new_data[data_len_train:])
        print(len(new_train_annot["images"]), "----", len(new_val_annot["images"]))
        return new_data   
    
    for key in grouped_data.keys():
        curr_group_im = grouped_data[key]["images"]
        curr_group_ann = grouped_data[key]["annotations"]
        
        data_p(curr_group_im)
        
        data_p(curr_group_ann, False)
    
    list_img_val = []
    list_ann_val = []    
    list_img_train = []
    list_ann_train = []
    for img in new_val_annot["images"]:
        for im in img:
            list_img_val.append(im)
            
    for img1 in new_train_annot["images"]:
        for im1 in img1:
            list_img_train.append(im1)
            
    for ann in new_val_annot["annotations"]:
        for an in ann:
            list_ann_val.append(an)
    
    
    for ann_a in new_train_annot["annotations"]:
        for aa in ann_a:
            list_ann_train.append(aa)
      
    new_train_annot["images"] = list_img_train
    new_train_annot["annotations"] = list_ann_train
    new_val_annot["images"] = list_img_val
    new_val_annot["annotations"] = list_ann_val
    with open("coco_all_train_new_final.json", "w") as f:
        json.dump(new_train_annot, f)
        
    with open("coco_all_val_new_final.json", "w") as g:
        json.dump(new_val_annot, g)
    
    
train_test_all("./coco_vids_all_ready_final.json")
    
