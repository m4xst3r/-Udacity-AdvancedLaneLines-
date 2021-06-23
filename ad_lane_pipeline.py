import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from gui_util import AdjustSobel
from gui_util import AdjustHLS

# Important constants and paremeters
cam_values = pickle.load(open('cam_values.p', "rb"))

images = glob.glob(r".\test_images\*.jpg")

def preprocess_image(img, cam_values):
    """
    Function to prepare image for lane extraction:
    1. undistort image
    2. crop a region of interest
    3. convert image to hsv
    4. Get binary gradient image using sobel
    5. transfer binary image using the roi

    Args:
        img ([type]): [description]
    """

    vertices = np.array([[
        ((image.shape[1]/2 - 80),image.shape[0]/1.6),
        ((image.shape[1]/2 + 80),image.shape[0]/1.6),
        ((image.shape[1] - 150  ),image.shape[0] - 55),
        (0 + 225,image.shape[0] - 55)]], dtype=np.int32)

    offset = vertices[0][0][0] - vertices[0][3][0]

    dst = np.array([[
        (125, 0),
        ((image.shape[1] -125),0),
        ((image.shape[1] - 265),image.shape[0]),
        (265,image.shape[0])]], dtype=np.int32)

    undist = cv2.undistort(img, cam_values['mtx'], cam_values['dist'], None, cam_values['mtx'])

    bin_image_thresh = abs_sobel_thresh(undist, orient='x', thresh_min = 10, thresh_max = 100)
    #if disturbance is to high reset to 150-170 (min value)
    bin_image_hls = svalues_mask(undist, s_min = 120, s_max = 255)

    #combine both bin images
    bin_image = np.zeros_like(bin_image_thresh)
    bin_image[(bin_image_thresh == 1) | (bin_image_hls == 1)] = 1

    bin_image_crop = region_of_interest(bin_image, vertices)

    src = np.float32([[vertices[0][0]],[vertices[0][1]], [vertices[0][2]],[vertices[0][3]]])

    bin_image_warped = warp_perpsective(bin_image_crop, src, np.float32(dst))

    return undist, bin_image_warped

def calculate_lines(img):
    """
    function calculates the lines using following workflow:
    1. get lane starts using histogram

    Args:
        img ([type]): [description]
    """

    result_img = get_lane_points(img)

    return result_img


def region_of_interest(img, vertices):
    """
    keeps only the image information from the defined vertices.

    Args:
        img ([type]): [description]
        vertices ([type]): array of integer points
    """
    #creat a black image mask
    mask = np.zeros_like(img)

    #check the color space of the image and create a mask of the color channels 
    if len(img.shape) > 2:
        channel_cnt = img.shape[2]
        color_mask = (255,) * channel_cnt
    else:
        color_mask = 255

    #fill the mask with color from in the define vertices area
    cv2.fillPoly(mask, vertices, color_mask)

    #returen image only where the mask is not 0
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def svalues_mask(img, s_min = 170, s_max = 255):
    """
    returns a binary image where the s-channel values of the hls color space
    match the threshhold

    Args:
        img ([type]): [description]
        s_min (int, optional): [description]. Defaults to 170.
        s_max (int, optional): [description]. Defaults to 255.
    """

    #convert image to hsv color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    #extract channel s from hls image
    s_channel = hls[:,:,2]

    #create a binary image where the therhshold matches the s-value
    bin_image = np.zeros_like(s_channel)
    bin_image[(s_channel >= s_min) & (s_channel <= s_max)] = 1

    return bin_image

def abs_sobel_thresh(img, orient='x', thresh_min = 0, thresh_max = 255):
    """
    applies the sobel function on the given orientation
    Using the given threshholds a binary image is built

    Args:
        img ([type]): [description]
        orient (str, optional): [description]. Defaults to 'x'.
        thresh_min (int, optional): [description]. Defaults to 0.
        thresh_max (int, optional): [description]. Defaults to 255.
    """
    # Convert image to gray to be able to use the sobel function
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    #apply sobel function
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)

    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    
    else:
        print('orientation is not defined')

    #take only the absolute values 
    abs_sobel = np.absolute(sobel)

    #normalize the sobel values to the rang 0 to 255
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    #create a binary image using the given threshholds
    bin_mask = np.zeros_like(scaled_sobel)
    bin_mask[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return bin_mask

def warp_perpsective(img, src, dst):
    """
    returns a warped image based on the given source and dst array

    Args:
        img ([type]): [description]
        src ([type]): [description]
        dst ([type]): [description]
    """
    #get trnform matrix 
    M = cv2.getPerspectiveTransform(src, dst)

    #warp image to top down view
    warped_image = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped_image

def get_lane_points(img):
    """
    1. Use a histogram to get the x position of the lines
    2. Use the sliding window algorithm to get the lane points

    Args:
        img ([type]): [description]
    """

    #crop image to use only the bottom half 
    bottom_half = img[img.shape[0]//2:,:]

    #sum all pixels in a vertical orientation
    histogram = np.sum(bottom_half, axis=0)

    out_img = np.dstack((img, img, img*255))
    #get the middel point from histogram and search for max 
    #left and right from that middel point
    midpoint = np.int32(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[0:640])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    #Hyperparemeters for sliding window algroithm
    #amount of windows used
    nwindows = 9
    #width of the windows
    margin = 100
    #minimum amount of picels which have to be in the window
    minpix = 50

    #calculate height of the window
    window_height = np.int32(img.shape[0]//nwindows)
    #Get all nonzeor pixels of the image and sort the to x and y
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    #get current psotion of the lane line
    leftx_current = leftx_base
    rightx_current = rightx_base

    #creat an empty list to store the picel indices of the lane
    left_lane_ind = []
    right_lane_ind = []

    for window in range(nwindows):
        #identify the window boundaries in x and y
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # #Only for debugging showing the windows
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 

        #get all indices of the pciels that are not zero inside the windows and append the to the land list
        good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
        (nonzero_x >= win_xleft_low) &  (nonzero_x < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & 
        (nonzero_x >= win_xright_low) &  (nonzero_x < win_xright_high)).nonzero()[0]

        left_lane_ind.append(good_left_inds)
        right_lane_ind.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.sum(nonzero_x[left_lane_ind[-1]])//len(nonzero_x[left_lane_ind[-1]]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.sum(nonzero_x[right_lane_ind[-1]])//len(nonzero_x[right_lane_ind[-1]]))

        #concenate arrays to get all the indices in one alist
        left_lane_ind = np.concatenate(left_lane_ind)
        right_lane_ind = np.concatenate(right_lane_ind)

        #extract lefta dn right pixels
        leftx = nonzero_x[left_lane_ind]
        lefty = nonzero_y[left_lane_ind]
        rightx = nonzero_x[left_lane_ind]
        righty = nonzero_y[left_lane_ind]

    return out_img, leftx, lefty, rightx, righty

def fit_polynom(img):
    """
    get form the found point of a lane a polyonmial function to display the line

    Args:
        img ([type]): [description]
    """


for image in images:   
    
    image = cv2.imread(image)
    image, bin_image = preprocess_image(image, cam_values)

    line_image = calculate_lines(bin_image)

    # cv2.imshow('image', bin_image)
    # cv2.waitKey(0)

    plt.imshow(line_image)
    plt.show()