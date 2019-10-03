from ImageRetrieval.Ranking import RankingSimilar
from definitions import QSD1_PATH, BBDD_PATH, K, QSD1_CORRESPONDANCE_FILE, QSD1_RESULTS_FILE
import pickle
import os
from Evaluation import RankingEvaluation

if __name__ == "__main__":

    """Do some sort of grid search among parameters"""

    """Load all candidates from BBDD pictures"""
    bbddd_candidates = []
    for file in os.listdir(BBDD_PATH):
        if file.endswith(".jpg"):
            bbddd_candidates.append(os.path.join(BBDD_PATH, file))

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

    print(correspondance_dict)
    """Instantiate ranking object that will be able to find the most similar K pictures from candidates given a query image"""
    ranking = RankingSimilar(bbddd_candidates, K)

    """Loop over all the query images in Q1 and compute the K most similar and the actual 
    ground truth expected value so that then we can evaluate with MAP the performance of the 
    process"""
    results_q1 = []
    actuals_q1 = []
    for (i, file_q1) in enumerate(sorted(files_q1), 0):
        similars = ranking.findKMostSimilar(file_q1, "hellinger")
        results_index = [index for index, path in similars]
        results_q1.append(results_index)
        actuals_q1.append([correspondance_dict[i]])

    print(RankingEvaluation.evaluateMAP(actuals_q1, results_q1))


    """Write results"""
    with open(QSD1_RESULTS_FILE, 'wb') as f:
        pickle.dump(results_q1, f)
