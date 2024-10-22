import cv2
import numpy as np
import torch
import os
import random
import re

def new_reel(dir):
    number_pattern = re.compile(r'\d+')  # Regex to capture digits in the subdirectory names
    max_number = -1  # Initialize the max number as -1

    # Loop through each subdirectory in the given directory
    for subdir in os.listdir(dir):
        # Find all the numbers in the subdir name using regex
        numbers = number_pattern.findall(subdir)
        if numbers:
            # Convert found numbers to integers and get the maximum one
            max_num_in_subdir = max(map(int, numbers))
            # Update the global maximum number
            max_number = max(max_number, max_num_in_subdir)
    
    return max_number  # Return the maximum number found

# Example usage of new_reel
#print(new_reel("/home/emil/sapiens/pose/demo/data/itw_videos"))


def data(dir_pth):
    """
        Args:
            -- dir_pth : str or PathLike
        Function:    
            -- Given the initial exercise directory path, 
                extract all exercise video paths : unsupervised task (optimal)
    """
    all_paths = []
    dir_pth = os.path.abspath(dir_pth)
    exercise_list_dir = os.listdir(dir_pth)
    for ex in exercise_list_dir:
        ex_path_full = os.path.join(dir_pth, ex)
        ex_vids_list = os.listdir(ex_path_full)
        for vid in ex_vids_list:
            vid_path = os.path.join(ex_path_full, vid)
            all_paths.append(vid_path)
            
    return all_paths

def process_random(all_data, frame_path, type = None):
    frame_path = os.path.abspath(frame_path)
    next_dir_suf = new_reel(frame_path) + 1
    new_reel_dir = os.path.join(frame_path, f"reel{next_dir_suf}")
    os.makedirs(new_reel_dir, exist_ok=True)
    if type is None: 
        type_list = all_data
    else:
        type_list = [x for x in all_data if type in x]
    
    video = random.choice(type_list) # randomly choose a vide of a type or from all
    video_capture = cv2.VideoCapture(video)
    success, frame = video_capture.read()
    frm_cnt = 0
    # getting all the frames and saving them
    while success:
        cv2.imwrite(os.path.join(new_reel_dir, f"0000{frm_cnt}.jpg"), frame)
        success, frame = video_capture.read()
        frm_cnt+=1
    video_capture.release()
        
all_videos = data("/home/emil/Keypoints/Sapiens/sapiens/extensions/exercise_data")
process_random(all_videos, "/home/emil/Keypoints/Sapiens/sapiens/pose/demo/data/itw_videos", "lat pulldown_13")
#image_dir = "/home/emil/sapiens/pose/demo/data/itw_videos/reel2"
#for img in os.listdir(image_dir):
   # img_p = cv2.imread(os.path.join(image_dir, img), cv2.IMREAD_COLOR) 
   # print(img_p.shape)     
            
        
    