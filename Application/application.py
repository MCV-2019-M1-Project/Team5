from ImageRetrieval.Ranking import RankingSimilar
from definitions import QSD1_PATH, QSD2_PATH, BBDD_PATH, K, QSD1_CORRESPONDANCE_FILE, \
    QSD1_RESULTS_FILE, QSD2_CORRESPONDANCE_FILE, QSD2_RESULTS_FILE, MASK_CORRESPONDANCE_RESULT_FILE
import pickle
import os
from Evaluation import RankingEvaluation
from ImageDescriptors import *
import cv2


def computeSearch(bbddd_candidates_files, query_files, correspondance_dict, masks=False):
    """Loop over all the query images and compute the K most similar and the actual
    ground truth expected value so that then we can evaluate with MAP the performance of the
    process"""
    ranking = RankingSimilar(bbddd_candidates_files, K)
    predictions = []
    actuals = []
    for (i, query_file) in enumerate(sorted(query_files), 0):
        if i in correspondance_dict:
            similars = ranking.findKMostSimilar(query_file, "hellinger", masks)
            results_index = [index for index, path in similars]
            print(correspondance_dict[i])
            print(results_index)
            predictions.append(results_index)
            actuals.append([correspondance_dict[i]])

    return actuals, predictions


def runRanking(bbdd_path, qs_path, qs_correspondance_path, qs_results_path, masks=False):
    """Load all candidates from BBDD pictures"""
    bbddd_candidates_files = []
    for file in os.listdir(bbdd_path):
        if file.endswith(".jpg"):
            bbddd_candidates_files.append(os.path.join(bbdd_path, file))

    """Load all query images"""
    query_files = []
    for file in os.listdir(qs_path):
        if file.endswith(".jpg"):
            path = os.path.join(qs_path, file)
            query_files.append(path)

    """Load all correspondances for Q and create a dictionary from it"""
    with open(qs_correspondance_path, 'rb') as f:
        correspondance = pickle.load(f)

    print(correspondance)
    correspondance_dict = {}
    for query_bbddd_tuple in correspondance:
        correspondance_dict.update(dict(query_bbddd_tuple))

    print(correspondance_dict)
    """Instantiate ranking object that will be able to find the most similar K pictures from candidates given a query image"""
    actuals, predictions = computeSearch(bbddd_candidates_files, query_files, correspondance_dict, masks)

    print(RankingEvaluation.evaluateMAP(actuals, predictions))

    """Write results"""
    with open(qs_results_path, 'wb') as f:
        pickle.dump(predictions, f)


def runMasksEvaluation(bbdd_path, qs_path, mask_correspondance_path):

    """Load all candidates from BBDD pictures"""
    bbddd_candidates = []
    for file in os.listdir(bbdd_path):
        if file.endswith(".jpg"):
            bbddd_candidates.append(os.path.join(bbdd_path, file))

    """Load all query images"""
    query_files = []
    for file in os.listdir(qs_path):
        if file.endswith(".jpg"):
            path = os.path.join(qs_path, file)
            query_files.append(path)

    mask_correspondance_dict = {}
    for (i, query_file) in enumerate(sorted(query_files), 0):
        hist = Histogram.MaskedHistogram(query_file)
        maskImg = hist.maskimg
        mask_path_dirname = os.path.dirname(query_file)
        mask_path_basename = os.path.splitext(os.path.basename(query_file))[0] + "_result.png"
        mask_path = mask_path_dirname + "/" + mask_path_basename
        print(maskImg)
        print(mask_path)
        cv2.imwrite(mask_path, maskImg)
        mask_correspondance_dict.update({query_file: mask_path_basename})

    """Write results"""
    with open(mask_correspondance_path, 'wb') as f:
        pickle.dump(mask_correspondance_dict, f)


if __name__ == "__main__":
    #runRanking(BBDD_PATH, QSD1_PATH, QSD1_CORRESPONDANCE_FILE, QSD1_RESULTS_FILE, False)
    #runRanking(BBDD_PATH, QSD2_PATH, QSD2_CORRESPONDANCE_FILE, QSD2_RESULTS_FILE, True)
    runMasksEvaluation(BBDD_PATH, QSD2_PATH, MASK_CORRESPONDANCE_RESULT_FILE)
