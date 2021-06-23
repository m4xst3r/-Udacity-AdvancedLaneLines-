import cv2
import numpy as np

class AdjustSobel:
    def __init__(self, image, orient='x', thresh_min = 0, thresh_max = 255):
        self.image = image
        self._orient = orient
        self._thresh_min = thresh_min
        self._thresh_max = thresh_max

        def onchangeMinThr(pos):
            self._thresh_min = pos
            self._thresh_min += self._thresh_min + 1       # make sure the filter size is odd
            self._render()

        def onchangeMaxThr(pos):
            self._thresh_max = pos
            self._thresh_max += self._thresh_max + 1       # make sure the filter size is odd
            self._render()

        cv2.namedWindow('sobel')

        cv2.createTrackbar('thresh_min', 'sobel', self._thresh_min, 255, onchangeMinThr)
        cv2.createTrackbar('thresh_max', 'sobel', self._thresh_max, 255, onchangeMaxThr)

        self._render()

        print ("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('sobel')

    def abs_sobel_thresh(self, img, orient='x', thresh_min = 0, thresh_max = 255):
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
        bin_mask[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 255

        return bin_mask

    def _render(self):
        self._sobel_image = self.abs_sobel_thresh(self.image, self._orient, self._thresh_min, self._thresh_max)
        print(self._thresh_min, self._thresh_max)
            
        cv2.imshow('sobel', self._sobel_image)


class AdjustHLS:
    def __init__(self, image, s_min = 170, s_max = 255):
        self.image = image
        self._thresh_min = s_min
        self._thresh_max = s_max

        def onchangeMinThr(pos):
            self._thresh_min = pos
            self._thresh_min += self._thresh_min + 1       # make sure the filter size is odd
            self._render()

        def onchangeMaxThr(pos):
            self._thresh_max = pos
            self._thresh_max += self._thresh_max + 1       # make sure the filter size is odd
            self._render()

        cv2.namedWindow('s-channel')

        cv2.createTrackbar('thresh_min', 's-channel', self._thresh_min, 255, onchangeMinThr)
        cv2.createTrackbar('thresh_max', 's-channel', self._thresh_max, 255, onchangeMaxThr)

        self._render()

        print ("Adjust the parameters as desired.  Hit any key to close.")

        cv2.waitKey(0)

        cv2.destroyWindow('channel')

    def svalues_mask(self, img, s_min = 170, s_max = 255):
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
        bin_image[(s_channel >= s_min) & (s_channel <= s_max)] = 255

        print(s_min, s_max)

        return bin_image

    def _render(self):
        self._sobel_image = self.svalues_mask(self.image, self._thresh_min, self._thresh_max)

            
        cv2.imshow('s-channel', self._sobel_image)
