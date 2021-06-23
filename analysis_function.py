import numpy as np
import matplotlib.pyplot as plt

def plot_hist(img):
    """
    generates a histogram of a binary image

    Args:
        img ([type]): [description]
    """

    #crop image to use only the bottom half 
    bottom_half = img[img.shape[0]//2:,:]

    #sum all pixels in a vertical orientation
    histogram = np.sum(bottom_half, axis=0)
    plt.plot(histogram)
