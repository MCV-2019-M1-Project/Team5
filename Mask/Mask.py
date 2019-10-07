import numpy as np

class MaskComputation:
    @staticmethod
    def mostFrequentValueInHistogramBasedMask(histogram, img):
        MARGIN_VALUES = 10

        def extractMostFrequentRGB(unmasked_histogram):
            max_indexes = []
            for channel in (0, 1, 2):
                max_value = 0
                index = 0
                for (idx, value) in enumerate(unmasked_histogram[channel][0], 0):
                    if value >= max_value:
                        max_value = value
                        index = idx
                max_indexes.append(index)

            return max_indexes
        max_rgb = extractMostFrequentRGB(histogram)
        im = np.asarray(img)
        mask = ((np.asarray(im[:, :, 0] <= (max_rgb[0] + MARGIN_VALUES)) & np.asarray(im[:, :, 0] >= (max_rgb[0] - MARGIN_VALUES)) & \
            np.asarray(im[:, :, 1] <= (max_rgb[1] + MARGIN_VALUES)) & np.asarray(im[:, :, 1] >= (max_rgb[1] - MARGIN_VALUES)) & \
            np.asarray(im[:, :, 2] <= (max_rgb[2] + MARGIN_VALUES)) & np.asarray(im[:, :, 2] >= (max_rgb[2] - MARGIN_VALUES))))
        return (~mask).astype(np.uint8)