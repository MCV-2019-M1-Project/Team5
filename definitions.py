import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
QSD1_PATH = ROOT_DIR + "/data/qsd1_w1/"
QSD2_PATH = ROOT_DIR + "/data/qsd2_w1/"
BBDD_PATH = ROOT_DIR + "/data/bbdd/"
QSD1_CORRESPONDANCE_FILE = QSD1_PATH + "gt_corresps.pkl"
QSD1_RESULTS_FILE = ROOT_DIR + "/qsd1_results.pkl"
K = 10

HIST_SIZE  = [256, 256, 256]
HIST_RANGE = [0, 256, 0, 256, 0, 256]  # the upper boundary is exclusive
CHANNELS   = [0, 1, 2]