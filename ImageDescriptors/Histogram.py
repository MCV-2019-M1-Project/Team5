import cv2
from definitions import CHANNELS, HIST_RANGE, HIST_SIZE

#TASK 1 is done here
class Histogram:

    def __init__(self, path, hsv=True):
        self.img = cv2.imread(path)
        if hsv:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        if self.img is None:
            raise Exception('Could not open or find the image:', path)
        self.mask = None

    def histogram(self):
        histogram = cv2.calcHist([self.img], CHANNELS, self.mask, HIST_SIZE, HIST_RANGE)
        return histogram
