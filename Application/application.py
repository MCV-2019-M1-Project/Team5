from ImageRetrieval.Ranking import RankingSimilar
from definitions import QSD1_PATH, BBDD_PATH, K, QSD1_CORRESPONDANCE_FILE
import pickle
import os
import ml_metrics as metrics


def evaluateMAP(actual, predicted):
    print(actual)
    print(predicted)
    return metrics.mapk(actual, predicted)


def evaluateAP(actual, predicted):
    return metrics.apk(actual, predicted)


if __name__ == "__main__":

    """Do some sort of grid search among parameters"""

    """Load all candidates from BBDD pictures"""
    bbddd_candidates = []
    for file in os.listdir(BBDD_PATH):
        if file.endswith(".jpg"):
            bbddd_candidates.append(os.path.join(BBDD_PATH, file))

    """Instantiate ranking object that will be able to find the most similar K pictures from candidates given a query image"""
    ranking = RankingSimilar(bbddd_candidates, K)

    """Load all images from Q1"""
    files_q1 = []
    for file in os.listdir(QSD1_PATH):
        if file.endswith(".jpg"):
            path = os.path.join(QSD1_PATH, file)
            files_q1.append(path)

    """Load all correspondances for QS1 and create a dictionary from it"""
    with open(QSD1_CORRESPONDANCE_FILE, 'rb') as f:
        correspondance = pickle.load(f)

    correspondance_dict = {}
    for query_bbddd_tuple in correspondance:
        correspondance_dict.update(dict(query_bbddd_tuple))

    """Loop over all the query images in Q1 and compute the K most similar and the actual 
    ground truth expected value so that then we can evaluate with MAP the performance of the 
    process"""
    results_q1 = []
    actuals_q1 = []
    for (i, file_q1) in enumerate(sorted(files_q1), 1):
        similars = ranking.findKMostSimilar(file_q1, "chi")
        results_index = [index for index, path in similars]
        results_q1.append(results_index)
        actuals_q1.append(correspondance_dict[i])

    evaluateMAP(results_q1, actuals_q1)
