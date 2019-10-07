from ImageRetrieval.Ranking import RankingSimilar
from definitions import *
import pickle
import os
from Evaluation import RankingEvaluation, MaskEvaluation
from ImageDescriptors import *
import cv2


def computeSearch(bbddd_candidates_files, query_files, correspondance_dict, masks=False, evaluate=False):
    """Loop over all the query images and compute the K most similar and the actual
    ground truth expected value so that then we can evaluate with MAP the performance of the
    process"""
    ranking = RankingSimilar(bbddd_candidates_files, K)
    predictions = []
    actuals = []
    for (i, query_file) in enumerate(sorted(query_files), 0):
        print(query_file)
        similars = ranking.findKMostSimilar(query_file, "hellinger", masks)
        results_index = [index for index, path in similars]
        print(results_index)
        predictions.append(results_index)
        if evaluate and i in correspondance_dict:
            actuals.append([correspondance_dict[i]])

    print(predictions)
    return actuals, predictions


def runRanking(bbdd_path, qs_path, qs_correspondance_path, qs_results_path, masks=False, evaluate=False):
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
    correspondance_dict = {}
    if evaluate:
        with open(qs_correspondance_path, 'rb') as f:
            correspondance = pickle.load(f)

        for query_bbddd_tuple in correspondance:
            correspondance_dict.update(dict(query_bbddd_tuple))

    """Instantiate ranking object that will be able to find the most similar K pictures from candidates given a query image"""
    actuals, predictions = computeSearch(bbddd_candidates_files, query_files, correspondance_dict, masks, evaluate)

    if evaluate:
        map_at_k = RankingEvaluation.evaluateMAP(actuals, predictions)
        print('Mean Average Precision: {:.2f}\n'.format(map_at_k))

    """Write results"""
    with open(qs_results_path, 'wb') as f:
        pickle.dump(predictions, f)


def runMasks(bbdd_path, qs_path, mask_correspondance_path, mask_store_path):
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
        mask_path_dirname = mask_store_path
        mask_path_basename = os.path.splitext(os.path.basename(query_file))[0] + ".png"
        mask_path = mask_path_dirname + "/" + mask_path_basename
        cv2.imwrite(mask_path, maskImg)
        mask_correspondance_dict.update({query_file: mask_path_basename})

    """Write results"""
    with open(mask_correspondance_path, 'wb') as f:
        pickle.dump(mask_correspondance_dict, f)


def runMasksEvaluation(qs_path, mask_store_path):
    annotation_masks_files = []
    predicted_masks_files = []
    for file in os.listdir(qs_path):
        if file.endswith(".png"):
            path = os.path.join(qs_path, file)
            annotation_masks_files.append(path)

    for file in os.listdir(mask_store_path):
        if file.endswith(".png"):
            path = os.path.join(mask_store_path, file)
            predicted_masks_files.append(path)

    pixelTP = 0
    pixelFN = 0
    pixelFP = 0
    pixelTN = 0
    for (annotation_file, prediction_file) in zip(sorted(annotation_masks_files), sorted(predicted_masks_files)):
        annotation_mask = cv2.imread(annotation_file)
        predicted_mask = cv2.imread(prediction_file)
        [localPixelTP, localPixelFP, localPixelFN, localPixelTN] = MaskEvaluation.performance_accumulation_pixel(
            predicted_mask,
            annotation_mask)
        pixelTP = pixelTP + localPixelTP
        pixelFP = pixelFP + localPixelFP
        pixelFN = pixelFN + localPixelFN
        pixelTN = pixelTN + localPixelTN

    [pixelPrecision, pixelAccuracy, pixelSpecificity, pixelSensitivity] = MaskEvaluation.performance_evaluation_pixel(
        pixelTP,
        pixelFP,
        pixelFN,
        pixelTN)

    pixelF1 = 0
    if (pixelPrecision + pixelSensitivity) != 0:
        pixelF1 = 2 * ((pixelPrecision * pixelSensitivity) / (pixelPrecision + pixelSensitivity))

    print('TPrecision: {:.2f}, Recall: {:.2f}, F1: {:.2f}\n'.format(pixelPrecision,
                                                                    pixelSensitivity,
                                                                    pixelF1))


if __name__ == "__main__":
    runRanking(BBDD_PATH, QSD1_PATH, QSD1_CORRESPONDANCE_FILE, QSD1_RESULTS_FILE, True, True)
    runRanking(BBDD_PATH, QSD2_PATH, QSD2_CORRESPONDANCE_FILE, QSD2_RESULTS_FILE, True, True)
    runMasks(BBDD_PATH, QSD2_PATH, MASK_CORRESPONDANCE_RESULT_FILE, MASK_STORE_PATH)
    runMasksEvaluation(QSD2_PATH, MASK_STORE_PATH)
