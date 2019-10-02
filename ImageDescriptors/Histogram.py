import cv2

HIST_SIZE = 256
HIST_RANGE = (0, 256)  # the upper boundary is exclusive


class Histogram:

    def __init__(self, path):
        self.planes = cv2.split(cv2.imread(path))
        self.mask = None
        if self.img is None:
            raise Exception('Could not open or find the image:', path)

    def histogram(self, channels):
        return cv2.calcHist([self.img], [channels], self.mask, [HIST_SIZE], HIST_RANGE)
