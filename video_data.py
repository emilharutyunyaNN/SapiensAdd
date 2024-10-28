import scipy.io
import json
import os
import shutil
import cv2
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
    print(new_dir)
    data = load_mat(mat_file)
    x = data['pos_img'][0]
    y = data['pos_img'][1]
    n_kp = len(data['pos_img'][0])
    frames = len(data['pos_img'][0][0])
    annotation_id = num
    
    frame = cv2.imread(os.path.join(new_dir, os.listdir(new_dir)[0]))
    h, w, _ = frame.shape
    for i, img in enumerate(os.listdir(new_dir)):
        #print(i)
        if os.path.isdir(img) or img =='.AppleDouble':
            continue
        img_ind = int(img.split(".")[0])
        img_id = int(str(rnd)+ str(int(img.split(".")[0])))
        print(img_id)
        image_info = {
                "file_name": os.path.join(new_dir, img),  # path to the image/frame
                "id": img_id,
                "height": h,
                "width": w
            }
        coco_data["images"].append(image_info)
        #keypnts
        keypoints_x = [(x_k[img_ind-1]) for x_k in x]
        
        keypoints_y = [(y_k[img_ind-1]) for y_k in y]
        keypoints = list(zip(keypoints_x, keypoints_y))
        keypoints = [(float(x), float(y)) for x, y in keypoints]
        print(keypoints)
        
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
            }
        coco_data["annotations"].append(annotation)
        num+=1
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
    files = os.listdir(dir)
    ann_n = 0
    rnd = 0
    for file in files:
        if file == ".AppleDouble":
            continue
        abs_f = os.path.abspath(os.path.join(dir, file))
        full_f = os.path.join(abs_f, "joint_positions.mat")
        ann_n = mat_2_coco(full_f,ann_n,coco_data,rnd)
        rnd+=1
        
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=4)
        
#process_all("/home/emil/Keypoints/Sapiens/video_data/joint_positions/brush_hair", "coco_brushhair.json")


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
