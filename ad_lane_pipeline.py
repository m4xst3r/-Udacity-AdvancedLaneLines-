import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob

# Important constants and paremeters
cam_values = pickle.load(open('cam_values.p', "rb"))

images = glob.glob(r".\test_images\*.jpg")

def preprocess_image(img, cam_values):
    """
    Function to prepare image for lane extraction:
    1. undistort image
    2. convert image to hsv

    Args:
        img ([type]): [description]
    """
    undist = cv2.undistort(img, cam_values['mtx'], cam_values['dist'], None, cam_values['mtx'])

    

    return undist

for image in images:   
    
    image = cv2.imread(image)
    image = preprocess_image(image, cam_values)

    plt.imshow(image)
    plt.show()