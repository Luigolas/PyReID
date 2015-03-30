# coding=utf-8
# !/usr/bin/env python3
import cv2
import numpy as np
import package.execution as execution

__author__ = 'luigolas'

HISTCMP_CORRE = cv2.HISTCMP_CORREL  # 0
HISTCMP_CHISQR = cv2.HISTCMP_CHISQR  # 1
HISTCMP_INTERSECT = cv2.HISTCMP_INTERSECT  # 2
HISTCMP_BHATTACHARYYA = cv2.HISTCMP_BHATTACHARYYA  # 3
HISTCMP_EUCLID = 4
method_names = ["CORRE", "CHISQ", "INTER", "BHATT", "EUCLI"]


class Comparator(object):
    def compare(self, ev1, ev2):
        """

        :param ev1:
        :param ev2:
        :raise NotImplementedError:
        """
        raise NotImplementedError("Please Implement compare _method")


class CompHistograms(Comparator):
    """
    Method to compare histograms
    :param comp_method:
    :raise AttributeError:
    """

    def __init__(self, comp_method, weights=None):

        if comp_method not in [0, 1, 2, 3, 4]:  # Not the best way to check
            raise AttributeError("Comparisson _method must be one of the predefined")
        self.method = comp_method
        self._weights = weights
        self.name = method_names[self.method]
        self.dict_name = {"Comparator": method_names[self.method]}

    def compare_by_index(self, hists1index, hists2index):
        """
        Look for values at global values (probeX and galleryY in package.execution
        :param hists1index:
        :param hists2index:
        :return:
        """
        hists1 = execution.probeX[hists1index]
        hists2 = execution.galleryY[hists2index]
        return self.compare(hists1, hists2)

    def compare(self, hists1, hists2):
        """

        :param hists1:
        :param hists2:
        :return:
        """
        #TODO Have to admit entries as numpy array

        CompHistograms._check_size_params(hists1, hists2, self._weights)

        if self._weights is None:
            weights = [1]
        else:
            weights = self._weights

        comp_val = []
        hist1 = hists1.reshape((hists1.shape[0] / 2, 2))
        hist2 = hists2.reshape((hists2.shape[0] / 2, 2))
        for h1, h2 in zip(hist1, hist2):
            if isinstance(h1, list):  # 2D case
                comp_val.append(sum([self.compareHist(h1_sub, h2_sub) for (h1_sub, h2_sub) in zip(h1, h2)]))
            else:  # 3D case
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
        #     raise TypeError("Weights parameter must be a list of values")
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
