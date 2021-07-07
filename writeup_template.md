## Writeup Advanced Lane Finding Project

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/cam_calib.png "Undistorted"
[image2]: ./output_images/undist.png "Road Transformed"
[image3]: ./output_images/bin_image_sobel.png "Gradient Example"
[image4]: ./output_images/bin_image_hls.png "HLS Example"
[image5]: ./output_images/bin_image.png "Combined Bin"
[image6]: ./output_images/bin_image_cropped.png "Cropped Image"
[image7]: ./output_images/bin_image_warp.png "Warped Image"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for the camera calibration is in a seperate python programm "./calib_cam.py" because this is the first step and can be done seperated to the normal image processing pipeline.

Firt of all I build a for loop iterating over all images inside the "./camera_cal/" folder. Using the function `get_points_chessboard` to receive image points and object point which belong to each other. Object points are genreated by the dimension of the chessboard in this project there are 9 corners in x and 6 in y defined. Z points are not defined because the chessboard is mounte on a plane wall which has no differnces in z. The image points (`corners`) are received be the function OpenCV function `cv2.findChessboardCorners` to do this the image is first tranformed into gray. In the end the function will return `objpoints`, `corners` and a bool variable to determine if the extraction of image point was succesful.

For every succesful extraction both opbject points and image points are added to a list. After all images are processed the image, objectPoints and imagePoints are used to calulate the camera matrix using the function `cv2.calibrateCamera()` which is needed to perform an undirstortion on the image. Applying the undistortion function from OpenCV `cv2.undistort()` approves the camera matrix is calculated right:

![alt text][image1]

The working camera matrix together with all other values is stored in a pickle file to be able to access it from othe python programs. Ther is no need to to perform this task more than once as the calibration will not change.

### Pipeline (single images)

The used pipeline is divided into three steps:

1. Preprocess the image to get a binary image with only neccesary informations
2. Calculate the lines an all other properties
3. Postprocess the image drawing all calculated information and parameters into the final image

Each step is divded in small subfunctions for each step this way it easier to debug and see changes in the performance

#### 1. Preprocess the Image

In total the preprocessing includes 5 steps:
1. undistort image
2. Get binary gradient image using sobel
3. Get binary image using colour threshhold (hls)
4. crop a region of interest (roi)
5. warp binary image using the roi

### 1. Undistort a image
First each image needs to be undistorted using the OpenCV function `cv2.undistort()` the values for the camera matrix are stored in a pickle and an input of the `preprocess_image()` function.
This is an example of an undisorted image:

![alt text][image2]

### 2. Get binary gradient image using sobel

To create a binary image only containing the lines first the gradient information is used to obtain all the lines facing vertically. The function `abs_sobel_thresh()` receives the image, orientation and threshholds needed. The outcome is a binary picture similar to this one:

![alt text][image3]

### 3. Get binary image using colour threshhold (hls)

In this step also binary image is created but this time using the colour space. Espescially the S (saturation) value of an HLS space. The function `svalues_mask()` uses the same input and returns also a binary image:

![alt text][image4]

For both binary outputs the thershholds are defined seperately by using a sliding window and observin when the most information with less noice is present in all the example pictures.
Both binary pictures are combined to one for furher processing:

![alt text][image5]

### 4. Crop a roi
The binary image is cropped to only keep the neccesary information. To crop the image early helps to find the right values for the binary image as the information to be considered is less. The result image will only contain the area with the lines infront of the car defines by a vertices:

![alt text][image6]

### 5. Warp binary image using the roi

In the last step the define roi is warped to get a birds eye view of the lanes. This is need for the calculation of the lanes afterwars. To warp the image there is the need to define soruce and destination points. The source points are already defined from the roi:

```python
vertices = np.array([[
    ((img.shape[1]/2 - 80),img.shape[0]/1.59),
    ((img.shape[1]/2 + 80),img.shape[0]/1.59),
    ((img.shape[1] - 150  ),img.shape[0] - 40),
    (0 + 225,img.shape[0] - 40)]], dtype=np.int32)
```

The destination points arr sperately defined and need to match the source to not distort the image:
```python
dst = np.array([[
        (95, 0),
        ((img.shape[1] -95),0),
        ((img.shape[1] - 265),img.shape[0]),
        (265,img.shape[0])]], dtype=np.int32)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 560, 490      | 95, 0        | 
| 720, 490      | 1185, 0      |
| 1130, 680     | 1015, 720    |
| 225, 680      | 265, 720     |

To confirm the value are good an image with straight lines is used to observe if the lanes are still straight after the transforming which is the case:

![alt text][image7]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
