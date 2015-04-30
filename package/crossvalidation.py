__author__ = 'luigolas'

from package.dataset import Dataset
import copy
import itertools
import os
import time
import gc
from package.feature_extractor import Histogram
from package.execution import Execution
from package.statistics import Statistics
from package.utilities import InitializationError
from package.image import CS_BGR
from package.preprocessing import BTF
import package.feature_matcher as feature_matcher
import pandas as pd
import numpy as np


class CrossValidation():
    def __init__(self, execution, num_validations=10, train_size=0, test_size=0.5):
        self.execution = execution
        self.statistics = []
        self.mean_stat = Statistics()
        self.num_validations = num_validations
        self.train_size, self.test_size = train_size, test_size
        self.dataframe = None

    # def add_execution(self, execution):
    #     if isinstance(execution, Execution):
    #         self.executions.append(execution)
    #     else:
    #         raise TypeError("Must be an Execution object")

    # def add_statistics(self, stat):
    #     if isinstance(stat, Statistics):
    #         self.statistics.append(stat)
    #     else:
    #         raise TypeError("Must be a Statistics object")

    # def add_post_ranker(self, post_ranker):
    #     if isinstance(post_ranker, PostRankOptimization):
    #         self.post_rankers.append(post_ranker)
    #     else:
    #         raise TypeError("Must be a PostRankOptimization object")

    # def set_statistics_comparative(self, rangeX_comp=True):
    #     if rangeX_comp:
    #         self._stats_comparative.append(self._rangeX_comparative)

    def add_execs_grabcut_btf_histograms(self, probe, gallery, id_regex=None, segmenter_iter=None, masks=None,
                                     colorspaces=None, binss=None, regions=None, region_name=None, weights=None,
                                     dimensions=None, compmethods=None, preprocessing=None, train_test_split=None):
        """

        :param probe:
        :param gallery:
        :param id_regex:
        :param segmenter_iter:
        :param masks:
        :param colorspaces:
        :param binss: can be a list of bins or a tuple, like in range. ex:
                      binss = [[40,40,40], [50,50,50]]
                      binss = (40, 51, 10) # Will generate same result
        :param regions:
        :param dimensions:
        :param compmethods:
        :return:
        """
        print("Generating executions...")
        if not segmenter_iter: segmenter_iter = [2]
        if not compmethods: compmethods = [feature_matcher.HISTCMP_BHATTACHARYYA]
        if not dimensions: dimensions = ["1D"]
        if not colorspaces: colorspaces = [CS_BGR]
        if not masks: raise InitializationError("Mask needed")
        # if not regions: regions = [None]
        if not preprocessing:
            preprocessing = [None]
        if not binss:
            binss = [[32, 32, 32]]
        elif isinstance(binss, tuple):
            binss_temp = []
            for r in range(*binss):
                binss_temp.append([r, r, r])
            binss = binss_temp
        if train_test_split is None:
            train_test_split = [[10, None]]
        # elif type(train_test_split) != list:
        #     train_test_split = [train_test_split]

        for gc_iter, mask, colorspace, bins, dimension, method, preproc, split in itertools.product(
                segmenter_iter, masks, colorspaces, binss, dimensions, compmethods, preprocessing, train_test_split):

            if preproc is None:
                btf = None
            else:
                btf = BTF(preproc)
            ex = Execution(Dataset(probe, gallery, split[0], split[1]), Grabcut(mask, gc_iter, CS_BGR), btf)

            if id_regex:
                ex.set_id_regex(id_regex)

            # bin = bins[0:len(Histogram.color_channels[colorspace])]
            ex.set_feature_extractor(
                Histogram(colorspace, bins, regions=regions, dimension=dimension, region_name=region_name))

            ex.set_feature_matcher(feature_matcher.HistogramsCompare(method, weights))
            self.executions.append(ex)
        print("%d executions generated" % len(self.executions))

    def run(self):
        self.dataframe = None
        # self._check_initialization()

        if self.train_size == 0:
            self.execution.dataset.generate_train_set(train_size=None, test_size=None)
            print("******** Execution ********")
            r_m = self.execution.run()
            print("Calculating Statistics")
            for val in range(self.num_validations):
                self.execution.dataset.generate_train_set(train_size=0, test_size=self.test_size)
                statistic = Statistics()
                statistic.run(self.execution.dataset, r_m)
                self.statistics.append(statistic)
        else:
            for val in range(self.num_validations):
                print("******** Execution %d of %d ********" % (val + 1, self.num_validations))
                self.execution.dataset.generate_train_set(train_size=self.train_size, test_size=self.test_size)
                r_m = self.execution.run()
                statistic = Statistics()
                statistic.run(self.execution.dataset, r_m)
                self.statistics.append(statistic)

        CMCs = np.asarray([stat.CMC for stat in self.statistics])
        self.mean_stat.CMC = np.sum(CMCs, axis=0) / float(self.num_validations)
        self.mean_stat.AUC = np.mean([stat.AUC for stat in self.statistics])
        mean_values = np.asarray([stat.mean_value for stat in self.statistics])
        self.mean_stat.mean_value = np.mean(mean_values)
        # Saving results: http://stackoverflow.com/a/19201448/3337586

    # def _check_initialization(self):
    #     if not self.executions:
    #         raise InitializationError
    #     if self.statistics:
    #         if len(self.executions) != len(self.statistics):
    #             if len(self.statistics) != 1:
    #                 raise InitializationError
    #             else:
    #                 for i in range(1, len(self.executions)):
    #                     self.statistics.append(copy.copy(self.statistics[0]))

    def dict_name(self):
        """

        :return:
        """
        name = {}
        name.update(self.execution.dict_name())
        name.update(self.mean_stat.dict_name())
        name.update({"NumValidations": self.num_validations})
        return name

    # def to_csv(self, path):
    #     self.dataframe.to_csv(path_or_buf=path, index=False, sep="\t", float_format='%.3f')
    #     # pd.DataFrame.to_csv(float_format="")

    def to_excel(self, path):
        """

        :param path:
        :return:
        """
        data = self.dict_name()
        dataframe = pd.DataFrame(data, columns=self.order_cols(list(data.keys())), index=[0])

        if os.path.isfile(path):
            df = pd.read_excel(path)
            df = pd.concat([df, dataframe])
            # df = df.merge(self.dataframe)
            df.to_excel(excel_writer=path, index=False, columns=self.order_cols(list(df.keys())), float_format='%.3f')
        else:
            dataframe.to_excel(excel_writer=path, index=False, float_format='%.3f')

    @staticmethod
    def order_cols(cols):
        """
        Format of columns to save in Dataframe
        :param cols:
        """
        ordered_cols = ["Probe", "Gallery", "Train", "Test", "Segmenter"]
        ordered_cols.extend(sorted([col for col in cols if "Seg" in col and col != "Segmenter"]))
        ordered_cols.extend(["Normalization"])
        ordered_cols.extend(sorted([col for col in cols if "Norm" in col and col != "Normalization"]))
        ordered_cols.extend(["BTF", "Regions"])
        ordered_cols.extend(sorted([col for col in cols if "Reg" in col and col != "Regions"]))
        ordered_cols.extend(["Map"])
        ordered_cols.extend(sorted([col for col in cols if "Map" in col and col != "Map"]))
        ordered_cols.extend(["Feature_Extractor"])
        ordered_cols.extend(sorted([col for col in cols if "Fe" in col and col != "Feature_Extractor"]))
        ordered_cols.extend(["FMatcher"])
        ordered_cols.extend(sorted([col for col in cols if "FM" in col and col != "FMatcher"]))
        ordered_cols.extend(sorted([col for col in cols if "Range" in col]))
        ordered_cols.extend(["AUC", "MeanValue", "NumValidations"])
        rest = sorted([item for item in cols if item not in ordered_cols])
        ordered_cols.extend(rest)
        return ordered_cols