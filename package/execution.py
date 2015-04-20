import sys

__author__ = 'luigolas'

import math
# import time
from package.utilities import InitializationError
# import re
import numpy as np
# import package.image as image
# import package.evaluator as evaluator
# from memory_profiler import profile
# import itertools
from package.dataset import Dataset
from sklearn.externals.joblib import Parallel, delayed

# For big improvements in multiprocesing
# probeX = None
# galleryY = None
probeXtest = None
probeXtrain = None
galleryYtest = None
galleryYtrain = None


def _parallel_transform(fe, im, mask):
    return fe.transform(im, mask)


def _parallel_compare(comp, i1, i2):
    return comp.compare(i1, i2)


class Execution():
    def __init__(self, dataset=None, preproc=None, feature_extractor=None, comparator=None,
                 post_ranker=None, train_split=None):

        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = Dataset()
        if train_split is not None:
            self.dataset.generate_train_set(train_split)

        self.preprocessing = preproc
        self.feature_extractor = feature_extractor
        self.comparator = comparator
        self.comparison_matrix = None
        self.ranking_matrix = None
        self.post_ranker = post_ranker

    def set_probe(self, folder):
        self.dataset.set_probe(folder)

    def set_gallery(self, folder):
        self.dataset.set_gallery(folder)

    def set_id_regex(self, regex):
        self.dataset.set_id_regex(regex)

    # def set_segmenter(self, segmenter):
    #     self.segmenter = segmenter

    def set_feature_extractor(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def set_comparator(self, comparator):
        self.comparator = comparator

    def set_preprocessing(self, preprocessing):
        if self.preprocessing is None:
            self.preprocessing = [preprocessing]
        else:
            self.preprocessing.append(preprocessing)

    def set_post_ranker(self, post_ranker):
        self.post_ranker = post_ranker

    def name(self):
        """
        example:
            P2_cam1_P2_cam2_Grabcut2OptimalMask_Histogram_IIP_[5, 5, 5]_6R_3D_BHATT
        :return:
        """
        # TODO Set unique and descriptive name
        name = "%s_%s_%s" % (
            self.dataset.name(), self.feature_extractor.name, self.comparator.name)
        return name

    def dict_name(self):
        """
        example:
        name = {"Probe": "P2_cam1", "Gallery": "P2_cam2", "Segment": "Grabcut", "SegIter": "2", "Mask": "OptimalMask",
                "Evaluator": "Histogram", "EvColorSpace": "IIP", "EvBins": "[5, 5, 5]", "EvDims": "3D", "Regions": "6R",
                "Comparator": "BHATT"}
        :return:
        """
        name = {}
        if self.preprocessing is not None:
            for preproc in self.preprocessing:
                name.update(preproc.dict_name())
        name.update(self.dataset.dict_name())
        # name.update(self.segmenter.dict_name)
        name.update(self.feature_extractor.dict_name)
        name.update(self.comparator.dict_name)
        return name

    def run(self):
        global probeX, galleryY, probeXtest, galleryYtest, probeXtrain, galleryYtrain

        if sys.gettrace() is None:
            n_jobs = -1
        else:
            n_jobs = 1

        self._check_initialization()

        print("Loading dataset images")
        self.dataset.load_images()

        print("--Preprocessing--")  # Requires Masks already calculated for BTF!!!!
        self._preprocess(n_jobs)

        # if self.dataset.probe.masks is None or self.dataset.gallery.masks is None:
        # print("Calculating Masks")  # TODO: Add option for not segmenting
        # self._calc_masks(multiprocessing)

        print("--Extracting Features--")
        self._transform_dataset(1)

        # Calculate Comparison matrix
        print("Calculating Comparison Matrix")
        self._calc_comparison_matrix(n_jobs)

        # Calculate Ranking matrix
        print("Calculating Ranking Matrix")
        self._calc_ranking_matrix()

        print("Execution Finished")

    def unload(self):
        global probeX, galleryY, probeXtest, galleryYtest
        self.dataset.unload()
        del self.dataset
        self.comparator.weigths = None
        del self.comparator
        self.feature_extractor.bins = None
        self.feature_extractor.regions = None
        del self.feature_extractor
        for preproc in self.preprocessing:
            del preproc
        del self.preprocessing
        del self.comparison_matrix
        del self.ranking_matrix
        probeX = None
        galleryY = None
        probeXtest = None
        galleryYtest = None

    def _check_initialization(self):
        if self.dataset.probe is None:
            raise InitializationError("probe not initialized")
        if self.dataset.gallery is None:
            raise InitializationError("gallery not initialized")
        if self.dataset.id_regex is None:
            # self.dataset.set_id_regex("P[1-6]_[0-9]{3}")
            raise InitializationError("id_regex not initialized")
        # if self.segmenter is None:
        #     raise InitializationError("segmenter not initialized")
        if self.feature_extractor is None:
            raise InitializationError("feature_extractor not initialized")
        if self.comparator is None:
            raise InitializationError("comparator not initialized")

    # def _calc_masks(self, n_jobs=-1):
    #     imgs = self.dataset.probe.images_train + self.dataset.probe.images_test
    #     imgs += self.dataset.gallery.images_train + self.dataset.gallery.images_test
    #     results = Parallel(n_jobs)(delayed(_parallel_calc_masks)(self.segmenter, im) for im in imgs)
    #     train_len = self.dataset.train_size
    #     test_len = self.dataset.test_size
    #     self.dataset.probe.masks_train = results[:train_len]
    #     self.dataset.probe.masks_test = results[train_len:train_len + test_len]
    #     self.dataset.gallery.masks_train = results[train_len + test_len:-test_len]
    #     self.dataset.gallery.masks_test = results[-test_len:]

    def _preprocess(self, n_jobs=1):
        if not self.preprocessing:
            return
        else:
            for preproc in self.preprocessing:
                preproc.preprocess_dataset(self.dataset, n_jobs)

    def _transform_dataset(self, n_jobs=-1):
        global probeXtrain, galleryYtrain, probeXtest, galleryYtest
        images = self.dataset.probe.images_train + self.dataset.probe.images_test
        images += self.dataset.gallery.images_train + self.dataset.gallery.images_test
        masks = self.dataset.probe.masks_train + self.dataset.probe.masks_test
        masks += self.dataset.gallery.masks_train + self.dataset.gallery.masks_test

        args = ((im, mask) for im, mask in zip(images, masks))

        results = Parallel(n_jobs)(delayed(_parallel_transform)(self.feature_extractor, im, mask) for im, mask in args)

        train_len = self.dataset.train_size
        test_len = self.dataset.test_size
        probeXtrain = np.asarray(results[:train_len])
        probeXtest = np.asarray(results[train_len:train_len + test_len])
        galleryYtrain = np.asarray(results[train_len + test_len:-test_len])
        galleryYtest = np.asarray(results[-test_len:])

    # @profile
    def _calc_comparison_matrix(self, n_jobs=-1):
        global probeXtest, galleryYtest

        args = ((elem1, elem2) for elem1 in probeXtest for elem2 in galleryYtest)

        results = Parallel(n_jobs)(delayed(_parallel_compare)(self.comparator, e1, e2) for e1, e2 in args)

        self.comparison_matrix = np.asarray(results, np.float32)

        size = math.sqrt(self.comparison_matrix.shape[0])
        self.comparison_matrix.shape = (size, size)

    def _calc_ranking_matrix(self):
        import package.comparator as Comparator
        # noinspection PyTypeChecker
        if self.comparator.method == Comparator.HISTCMP_CORRE or self.comparator.method == Comparator.HISTCMP_INTERSECT:
            # The biggest value, the better
            self.ranking_matrix = np.argsort(self.comparison_matrix, axis=1)[:, ::-1].astype(np.uint16)
            # Reverse order by axis 1

            # self.ranking_matrix = np.argsort(self.comparison_matrix[:, ::-1])
        else:  # The lower value, the better
            # self.ranking_matrix = np.argsort(self.comparison_matrix, axis=1)
            self.ranking_matrix = np.argsort(self.comparison_matrix).astype(np.uint16)
            # self.ranking_matrix = np.argsort(self.comparison_matrix, axis=1)

#
# def paralyze(*args):
#     return args[0][0](*args[0][1:][0])
#     pass




