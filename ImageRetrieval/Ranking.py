from ImageDescriptors import Histogram, Similarity
import heapq


def compare(img, hist):
    """TODO: Use different distances"""
    h = Histogram.Histogram(img)
    return Similarity.chisquared(hist, h)


class RankingSimilar:

    def __init__(self, query, candidates, k):
        self.query = query
        self.candidates = candidates
        self.k = k

    def findKMostSimilar(self):
        """TODO: Use different distances"""
        hquery = Histogram.Histogram(self.query)
        return heapq.nsmallest(self.k, self.candidates, key=lambda img: compare(img, hquery))
