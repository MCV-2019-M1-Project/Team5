import cv2
from definitions import CHANNELS, HIST_RANGE, HIST_SIZE
from Mask import MaskComputation

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
        '''TODO: Idea: Here have a factory for the kind of mapping with different inputs so that I can have the masks with which I will run the MaskedHistogram'''
        self.mask = MaskComputation.mostFrequentValueInHistogramBasedMask(self.histogram(), self.img)



