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
[image8]: ./output_images/lane_fit.png "Lane Fit"
[image9]: ./output_images/result.png "Result"
[video1]: ./project_video_detected.mp4 "Video"

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

Each step is divded in small subfunctions for each step this way it easier to debug and see changes in the performance. All the steps are done in one program called "./ad_lane_pipeline.py".

#### 1. Preprocess the Image

In total the preprocessing includes 5 steps:
1. undistort image
2. Get binary gradient image using sobel
3. Get binary image using colour threshhold (hls)
4. crop a region of interest (roi)
5. warp binary image using the roi

##### 1. Undistort a image
First each image needs to be undistorted using the OpenCV function `cv2.undistort()` the values for the camera matrix are stored in a pickle and an input of the `preprocess_image()` function.
This is an example of an undisorted image:

![alt text][image2]

##### 2. Get binary gradient image using sobel

To create a binary image only containing the lines first the gradient information is used to obtain all the lines facing vertically. The function `abs_sobel_thresh()` receives the image, orientation and threshholds needed. The outcome is a binary picture similar to this one:

![alt text][image3]

##### 3. Get binary image using colour threshhold (hls)

In this step also binary image is created but this time using the colour space. Espescially the S (saturation) value of an HLS space. The function `svalues_mask()` uses the same input and returns also a binary image:

![alt text][image4]

For both binary outputs the thershholds are defined seperately by using a sliding window and observin when the most information with less noice is present in all the example pictures.
Both binary pictures are combined to one for furher processing:

![alt text][image5]

##### 4. Crop a roi
The binary image is cropped to only keep the neccesary information. To crop the image early helps to find the right values for the binary image as the information to be considered is less. The result image will only contain the area with the lines infront of the car defines by a vertices:

![alt text][image6]

##### 5. Warp binary image using the roi

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

#### 2. Calculate lanes and other properties

After the binary image is correctly extracted the lance calculation is done. In this step it neccesary to distingiush in which state the detection is to be able to use different algorithms and apply smoothing if possible.
Three states are defined based on the confidence of the detction:
1. Initial state: First Frame or for some frames there is no confident detection
2. Tracking State: Last three frames are confident
3. Smoothing Stated: Last five frames are confident

The pipeline starts in the initial state to process the first frame in this state the lines are extracted for the first time and therefore no information is known. The workflow to find the lanes is as follows:
1. Use a histogram to determine the x position of the lanes based on the maximum of the histogram
2. Start from the received x position and receive with a sliding window algorithm the indices of the lane pixels
3. In the last point the pixels are used to get the polynomial coeficients whith which it is possible to generate a function describing the lane
In the following example image shows the bin image with the points which are used to get the polynomial coefficients marked as well the resultin lane plotted.

![alt text][image8]

For a better handling all lane properties are stored in a class to access them with different functions and be able to store the vlaues. The class is defined in a seprate python programm called `ad_lane.py`. 
After the polynomial coefficients are found it is possible to also measure the curvature and the position of the car relative to the lane middle. This is done in two seperate functions for the curve `measure_curv()` and for the position `calc_veh_pos()`. 
First the curve is calculated at a specific point most likely as near to the car as possible because it is much more importan to know what happens next. This point is determined by using the bottom image point. To calculate the curvature the point is inserted in the curvature function  `Rcurve​=(1+(2Ay+B)^2)^3/2​/∣2A∣` with the values A and B corresponding to the polynomial coefficients `y=A*y^2+B*y+C` and the y value matching the point value. Using the values receifed before the function would give a curvature as a pixel value but the value in meters is needed. Pixel values are converted to meter by using conversion parameters these are defined manually using the image lane pixeal distancces and the know how that the lanes are 3 meter long and 3.7 meters wide:
```python
ym_per_pix = 30/720
xm_per_pix = 3.7/600
```
With the conversion parameter first the coefficient parameters are recalculated and together instered in the curvature function:
```python
cofA_left = xm_per_pix / (ym_per_pix**2) * left_fit[0]
cofB_left = (xm_per_pix/ym_per_pix) * left_fit[1]
left_curv_m = ((1+(2*cofA_left*y_curve*ym_per_pix+cofB_left)**2)**(2/2))/np.absolute(2*cofA_left)
```
After the curvature of both lanes is calulated the mean value between them is calculated to determine the curvature in the middle of the picture.

The position calculation uses a similar method but instead of calculating the curvature of the lane the position of the lanes at the bottom of the picture is calulates using the polynom `y=A*y^2+B*y+C` and of couse recalculated in meters with the same method mentioned in the curvature funtcion. The vehicle position is just determined by the middle of the picture the difference between the car position and the middle point of the two lanes determines the orientation, if the value is greater 0 its right otherwise the car is left of the road and the differnece determines how far away the car is from the middle lane. In the end the lines are drawn in the source image and both the car position and curvature are displayed in the top left of the picture. 

![alt text][image9]


If the first coefficients are found it is possible to reduce the effort and search only around the lane found in the previous picture. This is done in the function `search_with_poly()` by using the coefficients to get lanes for left and right and extract all non zero pixels within an area around the lines. With the pixels the new polynom coefficents are than calculated. The only thing which needs to be considered first before the function `search_with_poly()` can be used is to make sure the lanes in the previous frame are fine by a sanity check. The sanity check is comparing line width, parallelism and curvature is okay if this is the case than the sanity check is true and after three good sanity checks the polynom search can be used instead of using the histogram together with the sliding window. As long as the sanity check is fine the pipeline will always use the polynom search but if the sanity check is wrong it will get back to the sliding window method.
The sanity check itself uses the calculated lines and a given tolerance. The tolerance is determined by testing the performance on the video. The goal is to use tolerances which are not to accurate but also not to slight this is a very chellenging task.

To have an even smoother lane detetection after four consecutive frames which passed the sanity check additionally an average polynom is calculated of the last found polynoms. This smoothing will be continued till the sanity check fails which clears also the history to prevent using wrong polynoms in the calulation. 

### Pipeline (video)

Combining all the calulation mentioned in the chapter befor provides the output visualized in the following video:

Here's a [link to my video result](./project_video_detected.mp4)

![alt text][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

During the development one of the most cirtical tasks was to find the right parameters. Slight changes has a huge impact and still in difficult situations like shadows etc. the algorithms are not able to handle it proper. To solve this more and different filtering is necessary to get rid of noise. 
Additionaly the used algorithm are not that precise as the constants for instance to calculate from pixe to meter are fixed and not measured. It would be better to calculate these values with other sensors or measrue them preciseley.
The smoothing helped a lot to make the algorithm more stable but it should be adjusted to the speed of the car to realy detrmine how many frame should be used for smoothing.
For me one of the biggest problems was to eloberate the sanity checks as there is already a high tolerance in the detection the sanity checks need even higher tolerances. The sanity check should be programmed with the help of some measurements and should be used as a ground truth.
In the end to improve the algorithm more data/measurements are needed which can be used to improve and add algorithms.