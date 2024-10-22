import re
import os
import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib import image 
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
import matplotlib.cm as cm



def extract_keypoints_from_lines(lines, key_pnt = True):
    # Combine the lines into a single string
    
    combined_str = " ".join([line.strip() for line in lines])
    if 'bboxes' in combined_str:
        combined_str = combined_str.replace("array([[", "")
        combined_str = combined_str.replace("float32", "")
        keypoint_values = re.findall(r'[-+]?\d*\.\d+|\d+(?![a-zA-Z])', combined_str)
        #print(combined_str)
    else:
    # Remove the 'array([[[', if present, from the combined string
        combined_str = combined_str.replace("array([[[", "")
        #if "keypoints:" in combined_str:
            #print(combined_str)
    # Updated regex to capture both scientific notation and regular floating point numbers
    
        keypoint_values = re.findall(r'[-+]?\d*\.\d+e[+-]?\d+|[-+]?\d*\.\d+', combined_str)
    #if 'bboxes' in combined_str:
        #print("--")
        #print(keypoint_values)
    # Convert the extracted strings to float
    keypoint_floats = list(map(float, keypoint_values))
    #if 'bboxes' in combined_str:
        #print(keypoint_floats)
    # Group the floats into pairs of (x, y)
    if key_pnt:
        keypoints = [(int(keypoint_floats[i]), int(keypoint_floats[i + 1])) for i in range(0, len(keypoint_floats), 2)]
    else:
        keypoints = keypoint_floats
    return keypoints

def get_preds(file_path, max_lines=10509, format="133"):
    if format == "133":
        kp_num = 132
        ksc_num = 27
    elif format == "17":
        kp_num = 16
        ksc_num=4
    else:
        raise(ValueError, "Unsupported format, use 133 or 17")
    abs_file = os.path.abspath(file_path)
    
    with open(abs_file) as file:
        read = file.readlines()

    # Restrict to max_lines if provided
    read = read[:max_lines]
    read = [rd.strip() for rd in read]
    
    img_paths = []
    keypoints = []
    keypoint_scores = []
    bboxes = []
    i = 0
    while i < len(read):
        line = read[i]
        if 'img_path' in line:
            img_paths.append(line.split(": ")[1].strip().strip("'"))
        
        if 'keypoints:' in line:
            current_keypoints = extract_keypoints_from_lines(read[i:i+kp_num+1])
            if current_keypoints:
                #print("+")
                keypoints.append(current_keypoints)
            i += kp_num  # Move the index forward by the number of keypoints lines
        #print("--- after kp: ", i)
        if 'keypoint_scores:' in line:
            #print("***", line)
            current_keypoint_scores = extract_keypoints_from_lines(read[i:i+ksc_num], False)
           # print("curr: ", current_keypoint_scores)
            if current_keypoint_scores:
                keypoint_scores.append(current_keypoint_scores)
            i += ksc_num  # Move the index forward by the number of keypoint score lines
        #print("---kp scores: ", i)
        if 'bboxes:' in line:
            #print("---")
            current_bbox = extract_keypoints_from_lines(read[i:i+1], False)
            if current_bbox:
                #if bboxes != [] and current_bbox == bboxes[-1]:
                  #  continue
                #print('smt')
                bboxes.append(current_bbox)
        #print("---bbox : ",i)
        i += 1  # Continue to the next line
        #print(f"Keypoints: for {i} -----", keypoints)
        #print(f"Keypoint score: for {i} -----", keypoint_scores)
        #print(f"images: for {i} -----", img_paths)
        #print(f"Bboxes: for {i} -----", bboxes)
        #if i>229:
          #  break
        # Output the latest img_path, keypoints, and keypoint_scores only when all three are populated
        #if img_paths and keypoints and keypoint_scores:
         #   print(img_paths[-1], "------", keypoints[-1], "------", keypoint_scores[-1])
        
    
    #bboxes = set(bboxes)
    #print(bboxes, len(bboxes))
    bboxes_no_duplicates = []
    for bbox in bboxes:
        if bbox not in bboxes_no_duplicates:
            bboxes_no_duplicates.append(bbox)
        #bboxes = [list(x) for x in set(tuple(x) for x in bboxes)]
    bboxes = bboxes_no_duplicates
    #print(bboxes, len(bboxes))

    print(len(img_paths), len(keypoints[0]), len(bboxes))
    return list(zip(img_paths, keypoints, keypoint_scores, bboxes))
# Call the function with your file path
#data = get_preds("/home/emil/Keypoints/Sapiens/sapiens/pose/metrics.txt")
data = get_preds("/home/emil/Keypoints/Sapiens/sapiens/pose/metrics_och.txt", max_lines=20733,format="17")
#print(data)
import matplotlib.pyplot as plt

import time
def plot(data, dir):
    def get_color_score(score):
        blue = int(score * 255.0)
        red = int((1 - score) * 255.0)
        return (red, 0, blue)
    
    print("----", len(data))
    print(data[24])
    for i, elem in enumerate(data):
        img = cv2.imread(elem[0])
        print("----", img.shape, "----", elem, "----", i)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        for j, (point, score) in enumerate(zip(elem[1], elem[2])):
            score_p = get_color_score(score)
            cv2.circle(img_rgb, point, radius=5, color=score_p, thickness=-1)
        
        
        bbox = elem[3]  # Example bbox format [xmin, ymin, xmax, ymax]
        cv2.rectangle(img_rgb, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 255, 0), thickness=2)
        # Convert image to RGB for matplotlib
        #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create a figure with two subplots: one for the image and one for the color bar
        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10, 1]}, figsize=(20, 12))
        
        # Show the image
        ax[0].imshow(img_rgb)
        ax[0].axis('off')  # Hide axes for the image
        
        # Create color bar
        norm = mcolors.Normalize(vmin=0, vmax=1)
        cmap = cm.ScalarMappable(norm=norm, cmap=cm.coolwarm_r)  # Blue to Red colormap
        cmap.set_array([])
        
        # Add color bar on the second subplot
        plt.colorbar(cmap, cax=ax[1], orientation='vertical', label='Confidence Score')
        
        plt.show()
        # Save the output
        
        fig.savefig(f"./{dir}/img_with_colorbar_{i}.jpg")
        plt.close()

plot(data, "och_imgs")