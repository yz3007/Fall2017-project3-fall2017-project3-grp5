import os
import numpy as np
from PIL import Image
import gist
import csv
import time
   
def feature_output(path):
    # Write headers for feature.csv file
    with open("feature.csv", "w") as feature_csv:
        writer = csv.writer(feature_csv, delimiter=',')
        writer.writerow(['Image Name'])
    jpg_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
    for index, jpg_pic in enumerate(jpg_files):
        
        feature = feature_extract(path, jpg_pic)
        # Record time
        # if index % 50 == 0:
        #     print(index)
        #     print("--- %s seconds ---" % (time.time() - start_time))
        with open("feature.csv", "a") as feature_csv:
            writer = csv.writer(feature_csv, delimiter=',')
            writer.writerow([jpg_pic]+feature)

# Feature extraction function for a picture
def feature_extract(path, jpg_pic):
    jpg_pic_path = path + "/" + jpg_pic
    pilimg = Image.open(jpg_pic_path)
    img = np.asarray(pilimg)
    if len(img.shape) == 2:
        img = greyToRGB(img)
    desc = gist.extract(img)
    feature = np.ndarray.tolist(desc)
    return(feature)

# For grayscale picture, change it to RGB picture
# because GIST descriptor is used for RGB pictures
def greyToRGB(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = img
    ret[:, :, 1] = ret[:, :, 2] = ret[:, :, 0]
    return(ret)



# Record time
# start_time = time.time() 
# feature_output(path = "/Users/siyi/Documents/Study-Columbia/17FALL/GR5243-Applied-Data-Science/Project3/training_set/images")
# Record time
# print("--- %s seconds ---" % (time.time() - start_time))



