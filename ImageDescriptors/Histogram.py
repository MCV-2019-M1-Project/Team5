import cv2
from definitions import CHANNELS, HIST_RANGE, HIST_SIZE
import numpy as np
import collections


# TASK 1 is done here
class Histogram:

    def __init__(self, path, hsv=False, normalize=False):
        """Load image from path"""
        self.img = cv2.imread(path)
        if self.img is None:
            raise Exception('Could not open or find the image:', path)
        if hsv:
            """If required, convert to HSV space"""
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        """TODO: For QS2 pictures compute mask here"""
        self.normalize = normalize
        self.mask = None

    def histogram(self):
        """Compute the histogram of the immage with the defined sizes and channels. Pass a mask which might be None"""
        histogram = cv2.calcHist([self.img], CHANNELS, self.mask, HIST_SIZE, HIST_RANGE)
        if self.normalize:
            """If required normalize the histogram"""
            cv2.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return histogram


#Task 5 is here
class MaskedHistogram(Histogram):

    def __init__(self, path, hsv=False, normalize=False):
        super().__init__(path, hsv, normalize)
        self.MARGIN_VALUES = 10
        unmasked_histogram = self.histogram()
        max_rgb = self.extractMostFrequentRGB(unmasked_histogram)
        im = np.asarray(self.img)
        mask = ((np.asarray(im[:, :, 0] <= (max_rgb[0] + self.MARGIN_VALUES)) & np.asarray(im[:, :, 0] >= (max_rgb[0] - self.MARGIN_VALUES)) & \
            np.asarray(im[:, :, 1] <= (max_rgb[1] + self.MARGIN_VALUES)) & np.asarray(im[:, :, 1] >= (max_rgb[1] - self.MARGIN_VALUES)) & \
            np.asarray(im[:, :, 2] <= (max_rgb[2] + self.MARGIN_VALUES)) & np.asarray(im[:, :, 2] >= (max_rgb[2] - self.MARGIN_VALUES))))

        self.mask    = (~mask).astype(np.uint8)
        self.maskimg = 255 * self.mask

    @staticmethod
    def extractMostFrequentRGB(unmasked_histogram):
        max_indexes = []
        for channel in (0, 1, 2):
            max_value = 0
            index = 0
            for (idx, value) in enumerate(unmasked_histogram[channel][0], 0):
                if value >= max_value:
                    max_value = value
                    index = idx
            max_indexes.append(index)

        return max_indexes

