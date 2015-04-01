__author__ = 'luigolas'
import itertools
import math
import cv2
import time
from package.utilities import InitializationError
from package.image_set import ImageSet
import package.app as app
import re
import numpy as np
import multiprocessing
import package.image as image
# import package.evaluator as evaluator
# from memory_profiler import profile
import itertools
from package.dataset import Dataset
from sklearn.externals.joblib import Parallel, delayed


__author__ = 'luigolas'

# For big improvements in multiprocesing
probeX = None
galleryY = None
probeXtest = None
# probeXtrain = None
galleryYtest = None
# galleryYtrain = None


class Execution():
    def __init__(self, dataset=None, segmenter=None, preproc=None, feature_extractor=None, comparator=None,
                 post_ranker=None, train_split=None):

        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = Dataset()
        if train_split is not None:
            self.dataset.generate_train_set(train_split)

        self.segmenter = segmenter
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

    def set_segmenter(self, segmenter):
        self.segmenter = segmenter

    def set_feature_extractor(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def set_comparator(self, comparator):
        self.comparator = comparator

    def set_preprocessing(self, preprocessing):
        self.preprocessing = preprocessing

    def set_post_ranker(self, post_ranker):
        self.post_ranker = post_ranker

    def name(self):
        """
        example:
            P2_cam1_P2_cam2_Grabcut2OptimalMask_Histogram_IIP_[5, 5, 5]_6R_3D_BHATT
        :return:
        """

        name = "%s_%s_%s_%s" % (
            self.dataset.name(), self.segmenter.name, self.feature_extractor.name, self.comparator.name)
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
        name.update(self.preprocessing.dict_name())
        name.update(self.dataset.dict_name())
        name.update(self.segmenter.dict_name)
        name.update(self.feature_extractor.dict_name)
        name.update(self.comparator.dict_name)
        return name

    def run(self):
        global probeX, galleryY, probeXtest, galleryYtest
        probeX = []
        galleryY = []

        self._check_initialization()

        if self.dataset.probe.masks is None or self.dataset.gallery.masks is None:
            print("Calculating Masks")
            self.dataset.calc_masks(self.segmenter)

        print("Preprocessing")  # Requires Masks already calculated
        self.dataset.preprocessed_probe = self._preprocess()

        print("Tranforming Dataset")
        self._transform_dataset()

        if self.dataset.test_indexes is not None:
            # probeXtrain = probeX[self.dataset.train_indexes]
            # galleryYtrain = galleryY[self.dataset.train_indexes]
            probeXtest = probeX[self.dataset.test_indexes]
            galleryYtest = galleryY[self.dataset.test_indexes]

        # Calculate Comparison matrix
        print("Calculating Comparison Matrix")
        self._calc_comparison_matrix()

        # Calculate Ranking matrix
        print("Calculating Ranking Matrix")
        self._calc_ranking_matrix()

        probeX = None
        galleryY = None
        probeXtest = None
        galleryYtest = None

    def unload(self):
        self.dataset.unload()
        del self.dataset
        self.comparator.weigths = None
        del self.comparator
        self.feature_extractor.bins = None
        self.feature_extractor.regions = None
        del self.feature_extractor
        self.segmenter._mask = None
        del self.segmenter
        del self.comparison_matrix
        del self.ranking_matrix

    def _check_initialization(self):
        if self.dataset.probe is None:
            raise InitializationError("probe not initialized")
        if self.dataset.gallery is None:
            raise InitializationError("gallery not initialized")
        if self.dataset.id_regex is None:
            self.dataset.set_id_regex("P[1-6]_[0-9]{3}")
            # raise InitializationError("id_regex not initialized")
        if self.segmenter is None:
            raise InitializationError("segmenter not initialized")
        if self.feature_extractor is None:
            raise InitializationError("feature_extractor not initialized")
        if self.comparator is None:
            raise InitializationError("comparator not initialized")

    def _preprocess(self):
        if not self.preprocessing:
            return
        else:
            return self.preprocessing.preprocess(self.dataset)

    def _transform_dataset(self, n_jobs=-1):
        global probeX, galleryY
        if self.dataset.preprocessed_probe is not None:
            probeX = self.dataset.preprocessed_probe
        else:
            probeX = self.dataset.probe

        galleryY = self.dataset.gallery.images

        args_probe = ((img_p, mask_p) for img_p, mask_p in zip(probeX, self.dataset.probe.masks))
        args_gallery = ((img_g, mask_g) for img_g, mask_g in zip(galleryY, self.dataset.gallery.masks))
        args = itertools.chain(args_probe, args_gallery)

        results = Parallel(n_jobs)(delayed(_parallel_transform)(self.feature_extractor, im, mask) for im, mask in args)
        probeX = np.asarray(results[:self.dataset.probe.len])
        galleryY = np.asarray(results[self.dataset.probe.len:])

    # @profile
    def _calc_comparison_matrix(self, n_jobs=-1):
        global probeX, galleryY, probeXtest, galleryYtest

        if probeXtest is not None:
            args = ((elem1, elem2) for elem1 in probeXtest for elem2 in galleryYtest)
        else:
            args = ((elem1, elem2) for elem1 in probeX for elem2 in galleryY)

        # args = ((index1, index2) for index1 in range(self.dataset.probe.len)
        #         for index2 in range(self.dataset.gallery.len))

        results = Parallel(n_jobs)(delayed(_parallel_compare)(self.comparator, e1, e2) for e1, e2 in args)

        self.comparison_matrix = np.asarray(results, np.float32)

        size = math.sqrt(self.comparison_matrix.shape[0])
        self.comparison_matrix.shape = (size, size)

        # self.comparison_matrix.shape = (self.dataset.probe.len, self.dataset.gallery.len)

    def _calc_ranking_matrix(self):
        import package.comparator as Comparator
        # noinspection PyTypeChecker
        if self.comparator.method == Comparator.HISTCMP_CORRE or self.comparator.method == Comparator.HISTCMP_INTERSECT:
            #  The biggest value, the better
            self.ranking_matrix = np.argsort(self.comparison_matrix, axis=1)[:, ::-1].astype(np.int16)
            # Reverse order by axis 1

            # self.ranking_matrix = np.argsort(self.comparison_matrix[:, ::-1])
        else:  # The lower value, the better
            # self.ranking_matrix = np.argsort(self.comparison_matrix, axis=1)
            self.ranking_matrix = np.argsort(self.comparison_matrix).astype(np.int16)
            # self.ranking_matrix = np.argsort(self.comparison_matrix, axis=1)


def paralyze(*args):
    return args[0][0](*args[0][1:][0])
    pass


def _parallel_transform(fe, im, mask):
    return fe.transform(im, mask)


def _parallel_compare(comp, i1, i2):
    # return comp.compare_by_index(i1, i2)
    return comp.compare(i1, i2)


