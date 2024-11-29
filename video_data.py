import scipy.io
import json
import os
import shutil
import cv2
BIG_NUM = 1000000


CLASS_DICT = {
    "brush_hair": 0,
    "catch": 1,
    "clap":2,
    "climb_stairs":3,
    "golf": 4,
    "jump":5,
    "kick_ball": 6,
    "pick": 7,
    "pour":8,
    "pullup":9,
    "push": 10,
    "run":11,
    "shoot_ball": 12,
    "shoot_bow": 13,
    "shoot_gun":14,
    "sit":15,
    "stand": 16,
    "swing_baseball":17,
    "throw": 18,
    "walk": 19,
    "wave":20,
    
}
def load_mat(file):
    abs_f = os.path.abspath(file)
    
    data = scipy.io.loadmat(abs_f)
    return data

def load_json(file):
    abs_f = os.path.abspath(file)
    
    with open(abs_f, "r") as f:
        data = json.load(f)
        
    return data
#dt = load_json("./coco_brushhair.json")
#print(dt['images'])
#data = load_mat("/home/emil/Keypoints/Sapiens/video_data/joint_positions/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0/joint_positions.mat")
#print(len(data['pos_img'][0][0]))

def mat_2_coco(mat_file, num, coco_data, rnd):
    abs_mat = os.path.abspath(mat_file) 
    vid = os.path.dirname(abs_mat)
    vid_cls = os.path.dirname(vid)
    vid_name = os.path.relpath(vid, vid_cls)
    ann_dir = os.path.dirname(vid_cls)
    vid_cls_nm = os.path.relpath(vid_cls,ann_dir)
    moth_dir = os.path.dirname(os.path.dirname(vid_cls))
    img_dir = "Rename_Images"
    new_dir = os.path.join(f"{moth_dir}/{img_dir}/{vid_cls_nm}/{vid_name}")
   # print(new_dir)
    data = load_mat(mat_file)
    x = data['pos_img'][0]
    print("x: ", len(x[0]))
    y = data['pos_img'][1]
    n_kp = len(data['pos_img'][0])
    frames = len(data['pos_img'][0][0])
    #if frames != 40:
      #  return num
    annotation_id = num
    print("PARAMS: ", mat_file, num, rnd)
    frame = cv2.imread(os.path.join(new_dir, os.listdir(new_dir)[0]))
    h, w, _ = frame.shape
    
    
    
    try:
        for i, img in enumerate(os.listdir(new_dir)):
            #print(i)
            
                print("img: ", img)
                if os.path.isdir(img) or img =='.AppleDouble' or img == '.DS_Store':
                    continue
                print("img: ", img)
                img_ind = int(img.split(".")[0])
                img_id = int(str(rnd*BIG_NUM+ int(img.split(".")[0])))
                #print(img_id)
                file_name = os.path.join(new_dir, img)
                image_info = {
                        "file_name": file_name,  # path to the image/frame
                        "id": img_id,
                        "height": h,
                        "width": w,
                        "class": CLASS_DICT[[dt for dt in CLASS_DICT.keys() if dt in file_name][0]]
                    }
                
                #keypnts
                print([len(x_k) for x_k in x])
                print(img_ind)
                keypoints_x = [(x_k[img_ind-1]) for x_k in x]
                
                keypoints_y = [(y_k[img_ind-1]) for y_k in y]
                keypoints = list(zip(keypoints_x, keypoints_y))
                keypoints = [(float(x), float(y)) for x, y in keypoints]
                #print(keypoints)
                
                #bboxes
                
                x_min = min(keypoints_x)
                y_min = min(keypoints_y)
                x_max = max(keypoints_x)
                y_max = max(keypoints_y)
                width = x_max - x_min
                height = y_max - y_min
                annotation = {
                        "id": num,
                        "image_id": img_id,
                        "category_id": 1,
                        "keypoints": keypoints,  # Assuming keypoints array in correct format
                        "num_keypoints": n_kp,
                        "bbox": [x_min, y_min, width, height],  # Optional: bounding box data
                        "iscrowd": 0,
                        "area": width*height,  # Optional: area of the bounding box
                        #"score": 1,
                        "class": image_info["class"]
                    }
                coco_data["images"].append(image_info)
                coco_data["annotations"].append(annotation)
                num+=1
            
    except:
        print("++")
    return num

        
       
#mat_2_coco("/home/emil/Keypoints/Sapiens/video_data/joint_positions/brush_hair/April_09_brush_hair_u_nm_np1_ba_goo_0/joint_positions.mat")

def process_all(dir, output_file):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", 
                          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
                          "left_wrist", "right_wrist", "left_hip", "right_hip", 
                          "left_knee", "right_knee", "left_ankle", "right_ankle"],
            "skeleton": [[1, 2], [1, 3], [2, 4], [3, 5], [1, 6], [1, 7], [6, 8], [7, 9], 
                         [8, 10], [9, 11], [6, 12], [7, 13], [12, 14], [13, 15], [14, 16], [15, 17]]
        }]
    }
    ann_n = 0
    rnd = 0
    all_dirs = [os.path.join(dir, dr) for dr in os.listdir(dir) if dr in CLASS_DICT.keys()]
    for dr in all_dirs:
        if ".AppleDouble" in dr or ".DS_Store" in dr:
            continue
        files = os.listdir(dr)
        overall = len(files)
        
        for file in files:
            if file == ".AppleDouble" or file == '.DS_Store':
                continue
            abs_f = os.path.abspath(os.path.join(dr, file))
            full_f = os.path.join(abs_f, "joint_positions.mat")
            ann_n = mat_2_coco(full_f,ann_n,coco_data,rnd)
            rnd+=1
    coco_data["images"] = sorted(coco_data["images"], key=lambda x: x["id"])
    coco_data["annotations"] = sorted(coco_data["annotations"], key=lambda x: x["image_id"])
    print("----", len(coco_data["images"]), len(coco_data["annotations"]))
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)
        
#process_all("/home/emil/Keypoints/Sapiens/video_data/joint_positions", "coco_vids_new.json")


def load_annot(ann):
    with open(ann, "r") as ann_f:
        data = json.load(ann_f)
        
    d_ims = [os.path.dirname(dt["file_name"]).split("/")[-1] for dt in data["images"]]
    d_ims_ann = [dt["image_id"] for dt in data["annotations"]]
    print(len(d_ims), len(d_ims_ann))
    
    
   
    dict_frames = dict()
    for d in d_ims:
        dict_frames[d] = 0
    for annot in data["images"]:
        dict_frames[os.path.dirname(annot["file_name"]).split("/")[-1]]+=1
        
    print(dict_frames.values())
    #for key in dict_frames.keys():
        #if dict_frames[key]>1:
            #print(key)
    
#load_annot("./coco_all_val_new_final.json")        
    

def check_ann(ann, bb_ann):
    with open(ann, "r") as anf:
        ann_d = json.load(anf)
    with open(bb_ann, "r") as anbbf:
        annbb_d = json.load(anbbf)
        
    img, img_id = ann_d['images'][1000]['file_name'], ann_d['images'][1000]['id']
    print(img, img_id)
    bbox = [d['bbox'] for d in annbb_d if d['image_id'] == img_id][0]
    print(bbox)
    image = cv2.imread(img)
    
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 255, 0), thickness=2)
    
    cv2.imwrite("./img.jpg", image)          

#check_ann("./coco_brushhair.json", "./person_detection_results/bbox_annotations.json")


video_dim = (1280, 720)
fps = 200
duration = 200.0
start_center = (0.4, 0.6)
end_center = (0.5, 0.5)
start_scale = 0.7
end_scale = 1.0

video_dim = (1280, 720)  # Set the resolution
fps = 24  # Adjust FPS as needed

def make_vids(dir, frames=4, fps=24, video_dim=(640, 480), frame_repeats=5):
    dir = os.path.abspath(dir)
    paths_in_dir = [os.path.join(dir, file) for file in os.listdir(dir) if file.endswith(('.png', '.jpg', '.jpeg'))]
    paths_in_dir.sort(key=lambda x: int(os.path.splitext(x)[0].split('_')[-1]))
    future_vids = []
    vid_nms = []
    os.makedirs(os.path.join(dir, 'vids'), exist_ok=True)
    #sprint(paths_in_dir)
    #return
    for i in range(0, len(paths_in_dir), frames):
        future_vids.append(paths_in_dir[i:i+frames])
        name = os.path.join(dir, 'vids', f'video{int(i / frames)}.mp4')
        vid_nms.append(name)
    
    for i, vid in enumerate(future_vids):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(vid_nms[i], fourcc, fps, video_dim)
        
        for j in range(len(vid)):
            image = cv2.imread(vid[j])
            if image is None:
                print(f"Warning: Couldn't read {vid[j]}, skipping.")
                continue
            
            # Resize the image to match the video dimensions
            resized_image = cv2.resize(image, video_dim)
            
            # Write the same frame multiple times to extend duration
            for _ in range(frame_repeats):
                video.write(resized_image)
        
        video.release()
    
    print("Videos created successfully.")

# Example usage
#make_vids("./imgs", fps=24, video_dim=(1280, 720), frame_repeats=15)

def process_json(annot, dir):
    dir = os.path.abspath(dir)
    file_scales = []
    list_joints = [os.path.join(dir, elem) for elem in os.listdir(dir)]
    for elem in list_joints:
        if ".AppleDouble" not in elem:
            data = load_mat(os.path.join(elem, "joint_positions.mat"))
            #print(elem.split("/")[-1], "----", )
            file_scales.append((elem.split("/")[-1], data["scale"][0][0]))
    
    json_abs = os.path.abspath(annot)
    with open(json_abs, "r") as f:
        data_annot = json.load(f)
        
        
    new_annot = dict()
    new_annot["images"] = data_annot["images"]
    new_annot["annotations"] = []   
    new_annot["categories"] = data_annot["categories"]
    for ann in data_annot['annotations']:
        imgid = ann["image_id"]
        keypoints = ann["keypoints"]
        scale_annot = [dt for dt in data_annot["images"] if dt["id"] ==imgid][0]
        scale_key = scale_annot["file_name"].split("/")[-2]
        h, w = scale_annot["height"], scale_annot["width"]
        scale = [file_scale[1] for file_scale in file_scales if file_scale[0] == scale_key][0]
        keypoints_x = [kp[0] for kp in keypoints]
        keypoints_y = [kp[1]for kp in keypoints]
        new_kpts_x = [(keypoint_x/scale) for keypoint_x in keypoints_x]
        new_kpts_y = [(keypoint_y/scale) for keypoint_y in keypoints_y]
        new_kpts = list(zip(new_kpts_x, new_kpts_y))
        
        x_min = min(new_kpts_x)
        y_min = min(new_kpts_y)
        x_max = max(new_kpts_x)
        y_max = max(new_kpts_y)
        width = x_max - x_min
        height = y_max - y_min
        new_annot["annotations"].append({
                "id": ann["id"],
                "image_id": imgid,
                "category_id": 1,
                "keypoints": new_kpts,  # Assuming keypoints array in correct format
                "num_keypoints": ann["num_keypoints"],
                "bbox": [x_min, y_min, width, height],  # Optional: bounding box data
                "iscrowd": 0,
                "area": width*height,  # Optional: area of the bounding box
                #"score": 1,
            })
        
    new_file = os.path.join(os.path.dirname(json_abs), "coco_brushhair_norm.json")
    with open(new_file, "w") as f:
        json.dump(new_annot, f, indent=4)
#process_json("/home/emil/Keypoints/Sapiens/Brushhair/annotations/coco_brushhair.json", "/home/emil/Keypoints/Sapiens/video_data/joint_positions/brush_hair")

import re

def look_at_annot(ann):
    with open(ann ,"r") as f:
        data = json.load(f)
        
    dict_files = dict()
    files = set([os.path.dirname(im["file_name"]) for im in data["images"]])
    classes = set([im["class"] for im in data["images"]])
    dct = dict()
    print(len(files))
    for c in classes:
        dct[c]= 0
    for im in data["images"]:
        dct[im["class"]]+=1
    print(dct)
    return
    
    for f in files:
        dict_files[f] = 0
        
    for im in data["images"]:
        for key in dict_files.keys():
            if key in im["file_name"]:
                dict_files[key]+=1
    bad_vids = []
    for key in dict_files.keys():
        if dict_files[key] < 10:
            bad_vids.append(key)
            print(key, dict_files[key])
    #return
    data["images"] = list(filter(lambda d: os.path.dirname(d["file_name"]) not in bad_vids, data["images"]))
    new_annot_list = []   
    all_img_ids = []
    for im in data["images"]:
        all_img_ids.append(im["id"])
        
    for annot in data["annotations"]:
        if annot["image_id"] in all_img_ids:
            new_annot_list.append(annot)
    data["annotations"] = new_annot_list
    
    with open("coco_vids_all_ready_final.json", "w") as f:
        json.dump(data, f)
    
    return
    for f in files:
       # print(type(f), f)
        list_f = 0
        ind_end = 0
        for i, im in enumerate(data["images"]):
            #if "emil" in str(im["file_name"]):
               # print("+")
            if str(f) in str(im["file_name"]):
                #print(f, "-----", im["file_name"])
                ind_end = i
                list_f+=1
                #print(ind_end)
                #break
        #print("----", list_f)
        if list_f%4!=0:
            ind = list_f-int(4*int(list_f/4))
            #print(ind)
            del data["images"][ind_end-ind+1:ind_end+1]
     
    new_annot_list = []   
    all_img_ids = []
    for im in data["images"]:
        all_img_ids.append(im["id"])
        
    for annot in data["annotations"]:
        if annot["image_id"] in all_img_ids:
            new_annot_list.append(annot)
    data["annotations"] = new_annot_list
                
    for c in classes:
        dct[c]= 0
    for im in data["images"]:
        dct[im["class"]]+=1
    print(dct)
    with open("coco_vids_all_ready.json", "w") as f:
        json.dump(data, f)
    
    
look_at_annot("/home/emil/Keypoints/Sapiens/jhmdb/annotations/coco_all_train_new_final.json")




#lst = [1,2,4,5,6,7,8,9,10]
#del lst[6-2+1:6+1]
#print(lst)
