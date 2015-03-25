import multiprocessing
import numpy
import cv2
import time
from package.image import CS_IIP, CS_BGR, CS_HSV, iip_max, iip_min, hsvmax, colorspace_name

__author__ = 'luigolas'


class FeatureExtractor(object):
    def transform(self, image, mask=None):
        """

        :param image:
        :param mask:
        :raise NotImplementedError:
        """
        raise NotImplementedError("Please Implement evaluate _method")


class Histogram(FeatureExtractor):
    ranges = {CS_BGR: [0, 255, 0, 255, 0, 255],
              CS_IIP: [80.86236, 140.3336, -17.9941, 18.1964, -73.6702, -42.93588],
              # CS_IIP: [iip_min, iip_max] * 3,
              CS_HSV: [0, hsvmax[0], 0, hsvmax[1]]}

    channels = {CS_BGR: [0, 1, 2],
                CS_IIP: [0, 1, 2],
                CS_HSV: [0, 1]}

    def __init__(self, color_space, bins=32, regions=None, dimension="1D", region_name=None):
        if color_space is None or not isinstance(color_space, int):
            raise AttributeError("Colorspace parameter must be valid")
        self._colorspace = color_space
        if type(bins) == int:
            if color_space == CS_BGR:
                bins = [bins] * 3
            elif color_space == CS_HSV:
                bins = [bins] * 2
            elif color_space == CS_IIP:
                bins = [bins] * 3
        if len(bins) != len(Histogram.channels[color_space]):
            raise IndexError("Size of bins incorrect for colorspace " + str(color_space))
        self._bins = bins
        self._regions = regions
        if not isinstance(dimension, str) or not (dimension == "1D" or dimension == "3D"):
            raise AttributeError("Dimension must be \"1D\" or \"3D\"")
        self._dimension = dimension
        if regions is None and region_name is not None:
            raise AttributeError("Region name must be None for region None")
        if region_name is None:
            self._region_name = str(regions)
        else:
            self._region_name = region_name

        self.name = "Histogram_%s_%s_%s_%s" % (colorspace_name[self._colorspace], self._bins, self._region_name,
                                               self._dimension)
        self.dict_name = {"Evaluator": "Histogram", "EvColorSpace": colorspace_name[self._colorspace],
                          "EvBins": str(self._bins), "Regions": self._region_name, "EvDim": self._dimension}

    def transform(self, image, mask=None, normalization=cv2.NORM_MINMAX):
        """

        :param image:
        :param mask:
        :return:
        """
        if self._colorspace and self._colorspace != image.colorspace:
            image = image.to_color_space(self._colorspace)

        if self._regions is None:
            regions = [[0, image.shape[0]]]
        else:
            regions = self._regions

        histograms = []

        for region in regions:
            # print("Histogram")
            hist = self.calcHistNormalized(image, mask, region, normalization)
            histograms.append(hist)
        return histograms

    # noinspection PyPep8Naming
    def calcHistNormalized(self, image, mask, region, normalization):
        if mask is not None:
            mask = mask[region[0]:region[1], :]
        image = image[region[0]:region[1], :]

        if self._dimension == "3D":
            hist = cv2.calcHist([image], Histogram.channels[image.colorspace],
                                mask, self._bins, Histogram.ranges[image.colorspace])
            hist = self.normalize_hist(hist, normalization)
        else:  # 1D case
            hist = []
            for channel in Histogram.channels[image.colorspace]:
                # print("1D")
                h = cv2.calcHist([image], [channel],
                                 mask, [self._bins[channel]],
                                 Histogram.ranges[image.colorspace][channel*2:channel*2+2])
                # print("Append")
                hist.append(self.normalize_hist(h, normalization))
                # In 1D case it can't be converted to numpy array as it might have different dimension (bins) sizes
        return hist


    @staticmethod
    def normalize_hist(histogram, normalization=cv2.NORM_MINMAX):
        """
        if histogram.max() != 0:  # Some cases when mask completely occlude region

        cv2.normalize(histogram, histogram, 0, 1, cv2.NORM_MINMAX, -1)  # Default
        cv2.normalize(histogram, histogram, 1, 0, cv2.NORM_L1)  # Divide all by "num pixel". Sum of all values equal 1
        cv2.normalize(histogram, histogram, 1, 0, cv2.NORM_INF)  # Divide all values by max

        Any of the three have the same effect. Will use NORM_MINMAX
        :param histogram:
        :param normalization: if None, no nomalization takes effect
        :return:
        """
        if normalization == cv2.NORM_MINMAX:
            cv2.normalize(histogram, histogram, 0, 1, cv2.NORM_MINMAX, -1)  # Default
        elif (normalization == cv2.NORM_L1) or (normalization == cv2.NORM_INF):
            cv2.normalize(histogram, histogram, 1, 0, normalization)

        return histogram