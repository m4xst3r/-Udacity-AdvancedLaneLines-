import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle

# read in all the images in the calibration folder
calib_images = glob.glob(".\camera_cal\*.jpg")

#define chess board parameters:
nx = 9
ny = 6

# Arrays to store image point and opbject points
imgpoints = []
objpoints = []

def get_points_chessboard(img, nx, ny):
    """
    returns the obj and img points from one chessboard image
    """
    #Genreate obj points based on the chessboar from (0,0) to (nx-1, ny-1)
    objp = np.zeros((nx*ny,3), np.float32)
    #np.mgrid cretes two arrays with 9x5 which are than merged together using T (transpose) and reshape. Only the first 2 columns of objp are replaced
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

    #convert Image to gray scale 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #get chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    return ret, objp, corners

def calc_cam_values(img, objpoints, imgpoints):
    """
    Calculates camera matrix etc. using the fucntio cv2.calibrateCamera
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[:2], None, None)

    return ret, mtx, dist, rvecs, tvecs

#Iterate thorugh images and extract there image points
for image_path in calib_images:
    
    image = cv2.imread(image_path)

    ret, objp, imgp = get_points_chessboard(image, nx, ny)

    if ret == True:
        imgpoints.append(imgp)
        objpoints.append(objp)

    else:
        print("image is not usable: ", image_path)

ret, mtx, dist, rvecs, tvecs = calc_cam_values(image, objpoints, imgpoints)

#write cam values into a dict
cam_values = { "mtx": mtx, "dist": dist,"rvecs": rvecs,"tvecs": tvecs}

#Save cam values in a pickle
pickle.dump(cam_values, open("cam_values.p", "wb"))