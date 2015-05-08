import numpy
import cv2
import time
from package.image import CS_IIP, CS_BGR, CS_HSV, iip_max, iip_min, hsvmax, colorspace_name
from sklearn.externals.joblib import Parallel, delayed
import numpy as np

__author__ = 'luigolas'


def _parallel_transform(fe, *args):
    return fe.extract(*args)


class FeatureExtractor(object):
    def extract_dataset(self, dataset, n_jobs, verbosity=2):
        """


        :param dataset:
        :param n_jobs:
        :param verbosity:
        :return:
        """
        raise NotImplementedError("Please Implement extract_dataset method")

    def extract(self, image, *args):
        """

        :param image:
        :param mask:
        :raise NotImplementedError:
        """
        raise NotImplementedError("Please Implement evaluate method")

    def dict_name(self):
        """

        :return:
        """
        raise NotImplementedError("Please Implement dict_name method")


class Histogram(FeatureExtractor):
    color_ranges = {CS_BGR: [0, 256, 0, 256, 0, 256],
                    CS_IIP: [80.86236, 140.3336, -17.9941, 18.1964, -73.6702, -42.93588],
                    # CS_IIP: [iip_min, iip_max] * 3,
                    CS_HSV: [0, hsvmax[0], 0, hsvmax[1], 0, hsvmax[2]]}

    # color_channels = {CS_BGR: [0, 1, 2],
    #             CS_IIP: [0, 1, 2],
    #             CS_HSV: [0, 1, 2]}

    def __init__(self, color_space, bins=None, dimension="1D"):
        if color_space is None or not isinstance(color_space, int):
            raise AttributeError("Colorspace parameter must be valid")
        self._colorspace = color_space
        if bins is None:
            bins = 32
        if type(bins) == int:
            if color_space == CS_BGR:
                bins = [bins] * 3
            elif color_space == CS_HSV:
                bins = [bins] * 2
            elif color_space == CS_IIP:
                bins = [bins] * 3

        self._channels = [i for i, e in enumerate(bins) if e != 0]
        self._original_bins = bins
        self._bins = [i for i in bins if i != 0]

        if not isinstance(dimension, str) or not (dimension == "1D" or dimension == "3D"):
            raise AttributeError("Dimension must be \"1D\" or \"3D\"")
        self._dimension = dimension

        self.name = "Histogram_%s_%s_%s" % (colorspace_name[self._colorspace], self._bins, self._dimension)

    def extract_dataset(self, dataset, n_jobs=-1, verbosity=2):
        if verbosity > 1: print("   Calculating Histograms %s, %s" % (colorspace_name[self._colorspace],
                                                                      str(self._original_bins)))
        images = dataset.probe.images_train + dataset.probe.images_test
        images += dataset.gallery.images_train + dataset.gallery.images_test
        images = dataset.probe.images_test
        images += dataset.gallery.images_test

        if dataset.probe.masks_test:
            masks = dataset.probe.masks_test
            masks += dataset.gallery.masks_test
        else:
            masks = [None] * (dataset.test_size) * 2

        if dataset.probe.regions_test:
            regions = dataset.probe.regions_test
            regions += dataset.gallery.regions_test
        else:
            regions = [None] * (dataset.test_size ) * 2

        if dataset.probe.maps_test:
            maps = dataset.probe.maps_test
            maps += dataset.gallery.maps_test
        else:
            maps = [None] * (dataset.test_size) * 2


        args = ((im, mask, region, m) for im, mask, region, m in zip(images, masks, regions, maps))

        results = Parallel(n_jobs)(delayed(_parallel_transform)(self, im, mask, reg, m) for im, mask, reg, m in args)

        # train_len = dataset.train_size
        test_len = dataset.test_size
        dataset.probe.fe_test = np.asarray(results[:test_len])
        dataset.gallery.fe_test = np.asarray(results[-test_len:])
        # dataset.probe.fe_train = np.asarray(results[:train_len])
        # dataset.probe.fe_test = np.asarray(results[train_len:train_len + test_len])
        # dataset.gallery.fe_train = np.asarray(results[train_len + test_len:-test_len])
        # dataset.gallery.fe_test = np.asarray(results[-test_len:])

    def extract(self, image, mask=None, regions=None, weights=None, normalization=cv2.NORM_MINMAX):
        """

        :param image:
        :param mask:
        :param regions:
        :param weights:
        :param normalization:
        :return:
        """

        if self._colorspace and self._colorspace != image.colorspace:
            # image = image.to_color_space(self._colorspace, normed=True)
            image = image.to_color_space(self._colorspace)

        if regions is None:
            regions = [[0, image.shape[0], 0, image.shape[1]]]

        histograms = []
        for region in regions:
            im = image[region[0]:region[1], region[2]:region[3]]

            if mask is not None and len(mask):
                m = mask[region[0]:region[1], region[2]:region[3]]
            else:
                m = mask

            if weights is not None and len(weights):
                weight = weights[region[0]:region[1], region[2]:region[3]]
            else:
                weight = 1

            hist = self.calcHistNormalized(im, m, weight, normalization)
            histograms.append(hist)

        return numpy.asarray(histograms).flatten().clip(0)

    # noinspection PyPep8Naming
    def calcHistNormalized(self, image, mask, weight, normalization):
        """


        :param image:
        :param mask:
        :param weight:
        :param normalization:
        :return:
        """
        # Using mask and map
        if mask is not None and len(mask):
            mask = mask * weight
        elif type(weight) != int:
            mask = weight
        else:
            mask = None

        # mask = weight                       # Using only MAP

        # mask = np.ones_like(mask)             # Not using mask nor map

        #  All comented, use only mask

        if self._dimension == "3D":
            # TODO add weighted ?
            # _channels = list(range(len(self._bins)))
            channels = numpy.nonzero(self._bins)

            hist = cv2.calcHist([image], channels,
                                mask, self._bins, Histogram.color_ranges[image.colorspace])
            # hist = self.normalize_hist(hist, normalization)
        else:  # 1D case
            hist = []
            for channel, bins in zip(self._channels, self._bins):
                if type(weight) != int:
                    h = calc_hist(image, channel, mask, bins,
                                  Histogram.color_ranges[image.colorspace][channel*2:channel*2+2])
                                  # [0., 1.])

                else:
                    h = cv2.calcHist([image], [channel], mask, [bins],
                                     Histogram.color_ranges[image.colorspace][channel*2:channel*2+2])
                                     # [0., 1.])

                # hist.extend(self.normalize_hist(h.astype(np.float32), normalization))
                hist.extend(h.astype(np.float32))
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

    def dict_name(self):
        return {"Feature_Extractor": "Histogram", "FeColorSpace": colorspace_name[self._colorspace],
                "FeBins": str(self._original_bins), "FeDim": self._dimension}


def calc_hist(im, ch, weight, bins, hist_range):
    """
    Inspiration: https://gist.github.com/nkeim/4455635
    Fast 1D histogram calculation. Must have all values initialized
    :param im: image to calculate histogram. Must be os shape (Height, Width, _channels)
    :param ch: channel to calculate histogram
    :param weight: weight map for values on histogram. Must be of shape (Height, Width) and float type
    :param bins: Numbers of bins for histogram
    :param hist_range: Range of values on selected channel
    :return:
    """
    # return np.bincount(np.digitize(im[:, :, ch].flatten(), np.linspace(hist_range[0], hist_range[1], bins + 1)[1:-1]),
    #                    weights=weight.flatten(), minlength=bins)
    return np.bincount(np.digitize(im[:, :, ch].flatten(), np.linspace(hist_range[0], hist_range[1], bins)[1:]),
                       weights=weight.flatten(), minlength=bins)
