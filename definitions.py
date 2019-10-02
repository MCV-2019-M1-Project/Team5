import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
QSD1_PATH = ROOT_DIR + "/data/qsd1_w1_small/"
QSD2_PATH = ROOT_DIR + "/data/qsd2_w1/"
BBDD_PATH = ROOT_DIR + "/data/bbdd/"
CORRESPONDANCE_FILE = ROOT_DIR + "/data/qsd1_w1_small/gt_corresps.pkl"
K = 10

HIST_SIZE  = [256, 256, 256]
HIST_RANGE = [0, 256, 0, 256, 0, 256]  # the upper boundary is exclusive
CHANNELS   = [0, 1, 2]