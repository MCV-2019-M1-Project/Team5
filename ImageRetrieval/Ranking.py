from ImageDescriptors.Histogram import *
from ImageDescriptors.Similarity import *
import heapq


# Task 3 is here
def compare(img, hist, method):
    """TODO: Use different distances"""
    h = Histogram(img).histogram()
    if method == "chi":
        similarity = chisquared(hist, h)
    elif method == "intersection":
        similarity = intersection(hist, h)
    elif method == "hellinger":
        similarity = hellinger(hist, h)
    else:
        raise Exception("Invalid compare method")

    return similarity


class RankingSimilar:

    def __init__(self, query, candidates, k):
        self.query = query
        self.candidates = enumerate(sorted(candidates), 1)
        self.k = k

    def findKMostSimilar(self, method):
        """TODO: Use different distances"""
        hquery = Histogram(self.query).histogram()
        return heapq.nsmallest(self.k,
                               self.candidates,
                               key=lambda index_image_tuple: compare(index_image_tuple[1], hquery, method))
