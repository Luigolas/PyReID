
__author__ = 'luigolas'

# import time
# import re
# from memory_profiler import profile
# import itertools
import sys
from package.utilities import InitializationError
import numpy as np
from package.dataset import Dataset


class Execution():
    def __init__(self, dataset=None, preproc=None, feature_extractor=None, feature_matcher=None,
                 post_ranker=None, train_split=None):

        if dataset is not None:
            assert(type(dataset) == Dataset)
            self.dataset = dataset
        else:
            self.dataset = Dataset()
        if train_split is not None:
            self.dataset.generate_train_set(train_split)

        self.preprocessing = preproc
        self.feature_extractor = feature_extractor
        self.feature_matcher = feature_matcher
        self.matching_matrix = None
        # self.ranking_matrix = None

    def set_probe(self, folder):
        self.dataset.set_probe(folder)

    def set_gallery(self, folder):
        self.dataset.set_gallery(folder)

    def set_id_regex(self, regex):
        self.dataset.set_id_regex(regex)

    def set_feature_extractor(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def set_feature_matcher(self, feature_matcher):
        self.feature_matcher = feature_matcher

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
            self.dataset.name(), self.feature_extractor.name, self.feature_matcher.name)
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
            for val, preproc in enumerate(self.preprocessing):
                # name.update(preproc.dict_name())
                # name.update({"Preproc%d" % val: preproc.dict_name()})
                preproc_name = preproc.dict_name()
                if preproc_name:
                    name.update({"Preproc%d" % val: preproc_name["name"]})
                    if "params" in preproc_name:
                        name.update({"Preproc%dParams" % val: preproc_name["params"]})
        name.update(self.dataset.dict_name())
        name.update(self.feature_extractor.dict_name())
        name.update(self.feature_matcher.dict_name())
        return name

    def run(self, verbosity=2, fe4train_set=False, njobs=-1):
        """

        :param verbosity:
        :param fe4train_set:
        :param njobs:
        :return:
        """
        if sys.gettrace() is None:
            n_jobs = njobs
        else:
            n_jobs = 1

        self._check_initialization()

        if verbosity > 0: print("Loading dataset images")
        self.dataset.load_images()

        if verbosity > 0: print("Preprocessing")
        self._preprocess(n_jobs, verbosity)

        if verbosity > 0: print("Extracting Features")
        self._extract_dataset(n_jobs, verbosity, fe4train_set)

        # Calculate Comparison matrix
        if verbosity > 0: print("Matching Features")
        self._calc_matching_matrix(n_jobs, verbosity)

        # Calculate Ranking matrix
        if verbosity > 0: print("Calculating Ranking Matrix")
        ranking_matrix = self._calc_ranking_matrix()

        if verbosity > 0: print("Execution Finished")
        return ranking_matrix

    def unload(self):
        global probeX, galleryY, probeXtest, galleryYtest
        self.dataset.unload()
        del self.dataset
        self.feature_matcher.weigths = None
        del self.feature_matcher
        self.feature_extractor.bins = None
        self.feature_extractor.regions = None
        del self.feature_extractor
        for preproc in self.preprocessing:
            del preproc
        del self.preprocessing
        del self.matching_matrix
        # del self.ranking_matrix
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
        if self.feature_extractor is None:
            raise InitializationError("feature_extractor not initialized")
        if self.feature_matcher is None:
            raise InitializationError("feature_matcher not initialized")

    def _preprocess(self, n_jobs=1, verbosity=2):
        if not self.preprocessing:
            return
        else:
            for preproc in self.preprocessing:
                preproc.preprocess_dataset(self.dataset, n_jobs, verbosity)

    def _extract_dataset(self, n_jobs=-1, verbosity=2, extract_train=False):
        self.feature_extractor.extract_dataset(self.dataset, n_jobs, verbosity, extract_train)

    def _calc_matching_matrix(self, n_jobs=-1, verbosity=2):
        self.matching_matrix = self.feature_matcher.match_probe_gallery(self.dataset.probe.fe_test,
                                                                        self.dataset.gallery.fe_test,
                                                                        n_jobs, verbosity)

    def _calc_ranking_matrix(self):
        import package.feature_matcher as Comparator
        if self.feature_matcher.method == Comparator.HISTCMP_CORRE or \
           self.feature_matcher.method == Comparator.HISTCMP_INTERSECT:
            # The biggest value, the better
            return np.argsort(self.matching_matrix, axis=1)[:, ::-1].astype(np.uint16)
            # Reverse order by axis 1
        else:  # The lower value, the better
            return np.argsort(self.matching_matrix).astype(np.uint16)



