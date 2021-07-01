import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
from gui_util import AdjustSobel
from gui_util import AdjustHLS
import ad_lane

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
        # cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        # (win_xleft_high,win_y_high),(0,255,0), 2) 
        # cv2.rectangle(out_img,(win_xright_low,win_y_low),
        # (win_xright_high,win_y_high),(0,255,0), 2) 

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
    rightx = nonzero_x[right_lane_ind]
    righty = nonzero_y[right_lane_ind]

    return leftx, lefty, rightx, righty

def fit_polynom(img, left_points_x, left_points_y, right_points_x, right_points_y):
    """
    get from the found points of a lane a polyonmial function to display the line

    Args:
        img ([type]): [description]
    """
    #fit a secong order polynom on the points
    left_fit = np.polyfit(left_points_y, left_points_x, 2)
    right_fit = np.polyfit(right_points_y, right_points_x, 2)

    #create x and y values for plotting
    plot_points = np.linspace(0, img.shape[0]-1, img.shape[0])

    return left_fit, right_fit, plot_points

def search_with_poly(img, left_fit, right_fit, plot_points):
    """
    seraches for a new polynom using a previous polynom

    Args:
        img ([type]): [description]
        left_fit ([type]): [description]
        right_fit ([type]): [description]

    Returns:
        [type]: [description]
    """
    #tolreance for the search 
    margin = 100

    #get all the nonzero pixels
    nonzero = img.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    #search for all the pixels which can be mapped to the polynom with +/- margin
    left_fit_line = left_fit[0]*nonzero_y**2 + left_fit[1]*nonzero_y + left_fit[2]
    left_lane_ind = ((nonzero_x >= (left_fit_line - margin)) & (nonzero_x <= (left_fit_line + margin)))
    right_fit_line = right_fit[0]*nonzero_y**2 + right_fit[1]*nonzero_y + right_fit[2]
    right_lane_ind = ((nonzero_x >= (right_fit_line - margin)) & (nonzero_x <= (right_fit_line + margin)))

    #get all the pixel based on indices
    leftx = nonzero_x[left_lane_ind]
    lefty = nonzero_y[left_lane_ind] 
    rightx = nonzero_x[right_lane_ind]
    righty = nonzero_y[right_lane_ind]

    #fit new polynom
    left_fit, right_fit, plot_points = fit_polynom(img, leftx, lefty, rightx, righty)

    return left_fit, right_fit, plot_points

def measure_curv(left_fit, right_fit, plot_points, ym_per_pix, xm_per_pix):
    """
    calculates the curvature using a given polynom

    Args:
        left_fit ([type]): [description]
        right_fit ([type]): [description]
        plot_points ([type]): [description]
    """

    #get the max y value (start of the lane) this is the place we want to calc the curvature
    y_curve = np.linspace(np.max(plot_points), 200, 200)

    #calculate/defin the new polynom values to get m instead of pixel

    cofA_left = xm_per_pix / (ym_per_pix**2) * left_fit[0]
    cofB_left = (xm_per_pix/ym_per_pix) * left_fit[1]

    cofA_right = xm_per_pix / (ym_per_pix**2) * right_fit[0]
    cofB_right = (xm_per_pix/ym_per_pix) * right_fit[1]

    #calculate the curvature using the formula: R = (1+(2Ay+B)^2)^3/2)/|2A| with y = A*y^2+B*y+C
    left_curv_m = ((1+(2*cofA_left*y_curve*ym_per_pix+cofB_left)**2)**(2/2))/np.absolute(2*cofA_left)
    right_curv_m = ((1+(2*cofA_right*y_curve*ym_per_pix+cofB_right)**2)**(2/2))/np.absolute(2*cofA_right)

    left_curv_m = np.mean(left_curv_m)
    right_curv_m = np.mean(right_curv_m)

    curv_mean = (left_curv_m + right_curv_m) / 2

    return curv_mean, left_curv_m, right_curv_m

def calc_veh_pos(img, left_fit, right_fit, plot_points, ym_per_pix,xm_per_pix):
    """
    calculate the vehicle position (middle point of the camera) between the two detecte lines (use polynom)
    """

    # calculate the position of the lines in m
    y_max = np.max(plot_points)

    #claulate coeeficient in m for polynom
    cofA_left = xm_per_pix / (ym_per_pix**2) * left_fit[0]
    cofB_left = (xm_per_pix/ym_per_pix) * left_fit[1]
    cofC_left = xm_per_pix * left_fit[2]

    cofA_right = xm_per_pix / (ym_per_pix**2) * right_fit[0]
    cofB_right = (xm_per_pix/ym_per_pix) * right_fit[1]
    cofC_right = xm_per_pix * right_fit[2]

    #calculate postition using the polynom function
    lane_pos_left = left_fit[0]*(y_max**2) + left_fit[1]*y_max + left_fit[2]
    lane_pos_right = right_fit[0]*(y_max**2) + right_fit[1]*y_max + right_fit[2]

    lane_pos_left_m = cofA_left*((y_max*ym_per_pix)**2) + cofB_left*(y_max*ym_per_pix) + cofC_left
    lane_pos_right_m = cofA_right*((y_max*ym_per_pix)**2) + cofB_right*(y_max*ym_per_pix) + cofC_right

    #calulate the middle of the two lines
    middle_pos = (lane_pos_left_m + lane_pos_right_m) / 2

    #calculate car position assuming the middel of the picture is the car
    car_pos = (img.shape[1] / 2) * xm_per_pix

    #calc diff between land middle pos and car pos
    diff_pos = car_pos - middle_pos

    if diff_pos >= 0:
        direction = "right"
    else:
        direction = "left"

    return np.absolute(diff_pos), direction


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
        ((img.shape[1]/2 - 80),img.shape[0]/1.6),
        ((img.shape[1]/2 + 80),img.shape[0]/1.6),
        ((img.shape[1] - 150  ),img.shape[0] - 40),
        (0 + 150,img.shape[0] - 40)]], dtype=np.int32)

    offset = vertices[0][0][0] - vertices[0][3][0]

    dst = np.array([[
        (220, 0),
        ((img.shape[1] - 220),0),
        ((img.shape[1] - 290),img.shape[0]),
        (290,img.shape[0])]], dtype=np.int32)

    undist = cv2.undistort(img, cam_values['mtx'], cam_values['dist'], None, cam_values['mtx'])

    bin_image_thresh = abs_sobel_thresh(undist, orient='x', thresh_min = 10, thresh_max = 100)
    #if disturbance is to high reset to 150-170 (min value)
    bin_image_hls = svalues_mask(undist, s_min = 150, s_max = 255)

    #combine both bin images
    bin_image = np.zeros_like(bin_image_thresh)
    bin_image[(bin_image_thresh == 1) | (bin_image_hls == 1)] = 1

    bin_image_crop = region_of_interest(bin_image, vertices)

    src = np.float32([[vertices[0][0]],[vertices[0][1]], [vertices[0][2]],[vertices[0][3]]])

    bin_image_warped = warp_perpsective(bin_image_crop, src, np.float32(dst))

    Minv = cv2.getPerspectiveTransform(np.float32(dst), src)

    return undist, bin_image_warped, Minv

def calculate_lines(img, lane):
    """
    function calculates the lines using following workflow:
    1. get lane starts using histogram

    Args:
        img ([type]): [description]
    """

    #Conversion parameters to calculate form pixels to m
    ym_per_pix = 30/720
    xm_per_pix = 3.7/600

    leftx, lefty, rightx, righty = get_lane_points(img)
    # left_fit, right_fit = fit_polynom(img, leftx, lefty, rightx, righty)

    if lane.detected == True:
        lane.left_fit, lane.right_fit, lane.plot_points = search_with_poly(img, lane.left_fit, lane.right_fit, lane.plot_points)
    else:
        lane.left_fit, lane.right_fit, lane.plot_points = fit_polynom(img, leftx, lefty, rightx, righty)


    lane.left_fit_line = lane.left_fit[0]*lane.plot_points**2 + lane.left_fit[1]*lane.plot_points + lane.left_fit[2]
    lane.right_fit_line = lane.right_fit[0]*lane.plot_points**2 + lane.right_fit[1]*lane.plot_points + lane.right_fit[2]


    # plt.plot(lane.left_fit_line, lane.plot_points, color='blue')
    # plt.plot(lane.right_fit_line, lane.plot_points, color='blue')

    # plt.imshow(img)
    # plt.show()

    lane.curv_radius, lane.left_curv, lane.right_curv = measure_curv(lane.left_fit, lane.right_fit, lane.plot_points, ym_per_pix, xm_per_pix)  
    lane.vehicle_pos, lane.vehicle_dir = calc_veh_pos(img, lane.left_fit, lane.right_fit, lane.plot_points, ym_per_pix,xm_per_pix)

    print (lane.sanity_check())
    if lane.sanity_check() == False:
        
        plt.plot(lane.left_fit_line, lane.plot_points, color='blue')
        plt.plot(lane.right_fit_line, lane.plot_points, color='blue')

        plt.imshow(img)
        plt.show()



def draw_lane(bin_image_warped, left_fit_line, right_fit_line, plot_points, Minv, img_shape):
    """
    This functions creates a overlay image to draw the lines

    Args:
        dst_img ([type]): [description]
    """

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(bin_image_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fit_line, plot_points]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_line, plot_points])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img_shape[1], img_shape[0])) 

    return newwarp

def post_process_image(warp_img, lane, Minv, undist_img):
    """
    uses the working image and the calculated values to show them in the final image

    Args:
        warp_img ([type]): image used to do the calculations
        lane ([type]): lane class with all neccesary values to draw
        Minv ([type]): Inverse matrix to recalulate pixel values
        undist_img ([type]): final image
    """

    lane_image = draw_lane(warp_img, lane.left_fit_line, lane.right_fit_line, lane.plot_points, Minv, undist_img.shape)
    result = cv2.addWeighted(image_undist, 1, lane_image, 0.3, 0)
    cv2.putText(result,'Radius of curvature=' + str(int(lane.curv_radius)) + 'm', (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)
    cv2.putText(result,'Vehicle is ' + "%.2f" %lane.vehicle_pos + 'm ' + lane.vehicle_dir + ' of the centre', (0,60), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),2)

    return result
    


# Important constants and paremeters
cam_values = pickle.load(open('cam_values.p', "rb"))

images = glob.glob(r".\test_images\*.jpg")
vid = cv2.VideoCapture('project_video.mp4')

lane = ad_lane.Lane()

while(vid.isOpened()):
    ret, frame = vid.read()

    if ret == True:
        image_undist, bin_image_warped, Minv = preprocess_image(frame, cam_values)

        calculate_lines(bin_image_warped, lane)

        result = post_process_image(bin_image_warped, lane, Minv, image_undist)

        cv2.imshow('image', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
	        break

    else:
        break

vid.release()
cv2.destroyAllWindows



#Only for writeup
# for image in images:   
    
#     image = cv2.imread(image)
#     image_undist, bin_image_warped, Minv = preprocess_image(image, cam_values)

#     calculate_lines(bin_image_warped, lane)

#     result = post_process_image(bin_image_warped, lane, Minv, image_undist)

#     cv2.imshow('image', result)
#     cv2.waitKey(0)