# coding=utf-8
# !/usr/bin/env python2
import cv2
import math
import numpy as np
from sklearn.externals.joblib import Parallel, delayed


__author__ = 'luigolas'

HISTCMP_CORRE = cv2.HISTCMP_CORREL  # 0
HISTCMP_CHISQR = cv2.HISTCMP_CHISQR  # 1
HISTCMP_INTERSECT = cv2.HISTCMP_INTERSECT  # 2
HISTCMP_BHATTACHARYYA = cv2.HISTCMP_BHATTACHARYYA  # 3
HISTCMP_EUCLID = 4
method_names = ["CORRE", "CHISQ", "INTER", "BHATT", "EUCLI"]


def _parallel_match(comp, i1, i2):
    return comp.match(i1, i2)


class FeatureMatcher(object):
    def match_dataset(self, dataset, n_jobs=-1):
        raise NotImplementedError("Please Implement compare _method")

    def match(self, ev1, ev2):
        """

        :param ev1:
        :param ev2:
        :raise NotImplementedError:
        """
        raise NotImplementedError("Please Implement compare _method")


class HistogramsCompare(FeatureMatcher):
    """
    Method to compare histograms
    :param comp_method:
    :raise AttributeError:
    """

    def __init__(self, comp_method, weights=None):
        if comp_method not in [0, 1, 2, 3, 4]:  # Not the best way to check
            raise AttributeError("Comparisson method must be one of the predefined for CompHistograms")
        self.method = comp_method
        self._weights = weights
        self.name = method_names[self.method]
        self.dict_name = {"Comparator": method_names[self.method]}

    def match_dataset(self, dataset, n_jobs=-1):
        print("   Comparing Histograms")
        args = ((elem1, elem2) for elem1 in dataset.probe.fe_test for elem2 in dataset.gallery.fe_test)

        results = Parallel(n_jobs)(delayed(_parallel_match)(self, e1, e2) for e1, e2 in args)

        comparison_matrix = np.asarray(results, np.float32)

        size = math.sqrt(comparison_matrix.shape[0])
        comparison_matrix.shape = (size, size)
        return comparison_matrix

    def match(self, hists1, hists2):
        """

        :param hists1:
        :param hists2:
        :return:
        """
        HistogramsCompare._check_size_params(hists1, hists2, self._weights)

        if self._weights is None:
            weights = [1]
        else:
            weights = self._weights

        comp_val = []
        num_histograms = len(weights)
        hist1 = hists1.reshape((num_histograms, hists1.shape[0] / num_histograms))
        hist2 = hists2.reshape((num_histograms, hists2.shape[0] / num_histograms))
        # hist1 = np.concatenate((np.asarray([0.] * 36, np.float32), hists1))
        # hist2 = np.concatenate((np.asarray([0.] * 36, np.float32), hists2))
        # hist1 = hist1.reshape((num_histograms, hist1.shape[0] / num_histograms))
        # hist2 = hist2.reshape((num_histograms, hist2.shape[0] / num_histograms))
        for h1, h2 in zip(hist1, hist2):
            if np.count_nonzero(h1) == 0 and np.count_nonzero(h2) == 0:
                # Might return inequality when both histograms are zero. So we compare two simple histogram to ensure
                # equality return value
                comp_val.append(self.compareHist(np.asarray([1], np.float32), np.asarray([1], np.float32)))
            else:
                comp_val.append(self.compareHist(h1, h2))
        comp_val = sum([i * j for i, j in zip(comp_val, weights)])
        return comp_val

    def compareHist(self, h1, h2):
        if self.method == HISTCMP_EUCLID:
            return np.linalg.norm(h1 - h2)
        else:
            return cv2.compareHist(h1, h2, self.method)

    @staticmethod
    def _check_size_params(hist1, hist2, weights):
        """

        :param hist:
        :param weights:
        :raise IndexError:
        """
        # if not isinstance(weights, list):
        # raise TypeError("Weights parameter must be a list of values")
        if weights:  # Both initialized
            if hist1.shape[0] % len(weights) != 0:
                raise IndexError("Size of hist and weights not compatible; hist: "
                                 + str(hist1.shape[0]) + "; weights: " + str(len(weights)))
            if hist2.shape[0] % len(weights) != 0:
                raise IndexError("Size of hist and weights not compatible; hist: "
                                 + str(hist2.shape[0]) + "; weights: " + str(len(weights)))
        elif hist1.shape[0] != hist2.shape[0]:
            raise IndexError(
                "Size of histograms must be the same. Size1:%d _ Size2:%d" & (hist1.shape[0], hist2.shape[0]))
