import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
QSD1_PATH = ROOT_DIR + "/data/qsd1_w1/"
QSD2_PATH = ROOT_DIR + "/data/qsd2_w1/"
QST1_PATH = ROOT_DIR + "/data/qst1_w1/"
QST2_PATH = ROOT_DIR + "/data/qst2_w1/"
BBDD_PATH = ROOT_DIR + "/data/bbdd/"
QSD1_CORRESPONDANCE_FILE = QSD1_PATH + "gt_corresps.pkl"
QSD2_CORRESPONDANCE_FILE = QSD2_PATH + "gt_corresps.pkl"
QSD1_RESULTS_FILE = ROOT_DIR + "/qsd1_results.pkl"
QSD2_RESULTS_FILE = ROOT_DIR + "/qsd2_results.pkl"
QST1_CORRESPONDANCE_FILE = QST1_PATH + "gt_corresps.pkl"
QST2_CORRESPONDANCE_FILE = QST2_PATH + "gt_corresps.pkl"
QST1_RESULTS_FILE = ROOT_DIR + "/qst1_results.pkl"
QST2_RESULTS_FILE = ROOT_DIR + "/qst2_results.pkl"
MASK_CORRESPONDANCE_RESULT_FILE = ROOT_DIR + "/mask_correspondance_result.pkl"
MASK_STORE_PATH = ROOT_DIR + "/data/masks_results/"
K = 10

HIST_SIZE  = [256, 256, 256]
HIST_RANGE = [0, 256, 0, 256, 0, 256]  # the upper boundary is exclusive
CHANNELS   = [0, 1, 2]