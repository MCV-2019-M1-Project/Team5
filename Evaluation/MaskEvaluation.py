import numpy as np

"""Code taken from https://github.com/MCV-2019-M1-Project/mcv-m1-code/blob/master/evaluation/evaluation_funcs.py"""


def performance_accumulation_pixel(pixel_candidates, pixel_annotation):
    """
    performance_accumulation_pixel()
    Function to compute different performance indicators
    (True Positive, False Positive, False Negative, True Negative)
    at the pixel level

    [pixelTP, pixelFP, pixelFN, pixelTN] = performance_accumulation_pixel(pixel_candidates, pixel_annotation)

    Parameter name      Value
    --------------      -----
    'pixel_candidates'   Binary image marking the foreground areas
    'pixel_annotation'   Binary image containing ground truth

    The function returns the number of True Positive (pixelTP), False Positive (pixelFP),
    False Negative (pixelFN) and True Negative (pixelTN) pixels in the image pixel_candidates
    """

    pixel_candidates = np.uint64(pixel_candidates > 0)
    pixel_annotation = np.uint64(pixel_annotation > 0)

    pixelTP = np.sum(pixel_candidates & pixel_annotation)
    pixelFP = np.sum(pixel_candidates & (pixel_annotation == 0))
    pixelFN = np.sum((pixel_candidates == 0) & pixel_annotation)
    pixelTN = np.sum((pixel_candidates == 0) & (pixel_annotation == 0))

    return [pixelTP, pixelFP, pixelFN, pixelTN]


def performance_evaluation_pixel(pixelTP, pixelFP, pixelFN, pixelTN):
    """
    performance_evaluation_pixel()
    Function to compute different performance indicators (Precision, accuracy,
    specificity, sensitivity) at the pixel level

    [pixelPrecision, pixelAccuracy, pixelSpecificity, pixelSensitivity] = PerformanceEvaluationPixel(pixelTP, pixelFP, pixelFN, pixelTN)

       Parameter name      Value
       --------------      -----
       'pixelTP'           Number of True  Positive pixels
       'pixelFP'           Number of False Positive pixels
       'pixelFN'           Number of False Negative pixels
       'pixelTN'           Number of True  Negative pixels

    The function returns the precision, accuracy, specificity and sensitivity
    """

    pixel_precision = 0
    pixel_accuracy = 0
    pixel_specificity = 0
    pixel_sensitivity = 0
    if (pixelTP + pixelFP) != 0:
        pixel_precision = float(pixelTP) / float(pixelTP + pixelFP)
    if (pixelTP + pixelFP + pixelFN + pixelTN) != 0:
        pixel_accuracy = float(pixelTP + pixelTN) / float(pixelTP + pixelFP + pixelFN + pixelTN)
    if (pixelTN + pixelFP):
        pixel_specificity = float(pixelTN) / float(pixelTN + pixelFP)
    if (pixelTP + pixelFN) != 0:
        pixel_sensitivity = float(pixelTP) / float(pixelTP + pixelFN)

    return [pixel_precision, pixel_accuracy, pixel_specificity, pixel_sensitivity]
