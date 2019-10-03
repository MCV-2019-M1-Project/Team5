from ImageDescriptors.Histogram import *
from ImageDescriptors.Similarity import *
import heapq

"""Define the Ranking method"""


# Task 3 is here
def comparePreComputed(candidatehist, queryhist, method):
    """Compute similarities """
    if method == "chi":
        similarity = chisquared(queryhist, candidatehist)
    elif method == "intersection":
        similarity = intersection(queryhist, candidatehist)
    elif method == "hellinger":
        similarity = hellinger(queryhist, candidatehist)
    else:
        raise Exception("Invalid compare method")

    return similarity


def compare(img, queryhist, method):
    """Used by the sorting algorithm receives the query histogram computed and on the fly compute the histogram for candidates and compute similarities """
    h = Histogram(img).histogram()
    return comparePreComputed(h, queryhist, method)


class RankingSimilar:

    def __init__(self, candidates, k, precompute=True):
        idx_images = enumerate(sorted(candidates), 0)
        self.precomputed = precompute
        if self.precomputed:
            self.candidates = []
            for (idx, img) in idx_images:
                self.candidates.append((idx, Histogram(img).histogram()))
        else:
            self.candidates = idx_images
        self.k = k

    def findKMostSimilar(self, queryImg, method):
        """Compute the histogram of the query image and run the search by using a Heap based algorithm to find the k
        images with a smallest distance where the comparing function is passed as the key with a lambda function"""
        hquery = Histogram(queryImg).histogram()

        if self.precomputed:
            f = lambda index_hist_tuple: comparePreComputed(index_hist_tuple[1], hquery, method)
        else:
            f = lambda index_image_tuple: compare(index_image_tuple[1], hquery, method)

        return heapq.nsmallest(self.k,
                               self.candidates,
                               key=f)
