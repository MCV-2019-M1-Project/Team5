import scipy.spatial.distance as dist
import cv2


# TASK 2 is here
def euclidean(h1, h2):
    return dist.euclidean(h1, h2)


def chisquared(h1, h2):
    return cv2.compareHist(h1, h2, method=cv2.HISTCMP_CHISQR)


def intersection(h1, h2):
    return cv2.compareHist(h1, h2, method=cv2.HISTCMP_INTERSECT)


def hellinger(h1, h2):
    return cv2.compareHist(h1, h2, method=cv2.HISTCMP_HELLINGER)
