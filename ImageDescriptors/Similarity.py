import scipy.spatial.distance as dist
import cv2


def euclidean(h1, h2):
    return dist.euclidean(h1, h2)


def chisquared(h1, h2):
    return cv2.compareHist(h1, h2, method="CV_COMP_CHISQR")


def intersection(h1, h2):
    return cv2.compareHist(h1, h2, method="CV_COMP_INTERSECT")


def hellinger(h1, h2):
    return cv2.compareHist(h1, h2, method="CV_COMP_HELLINGER")
