from package.post_ranker import PostRankOptimization

__author__ = 'luigolas'

from package.dataset import Dataset
import copy
from operator import itemgetter
import itertools
import os
import cv2
import time
import gc
from package.feature_extractor import Histogram
from package.image_set import ImageSet
from package.execution import Execution
from package.statistics import Statistics
from package.utilities import InitializationError
from package.image import CS_BGR
from package.preprocessing import BTF
import package.feature_matcher as comparator
import pandas as pd


class Configuration():
    def __init__(self):
        self.executions = []
        self.statistics = []
        self.post_rankers = []
        # self._stats_comparative = []
        self.dataframe = None

    def add_execution(self, execution):
        if isinstance(execution, Execution):
            self.executions.append(execution)
        else:
            raise TypeError("Must be an Execution object")

    def add_statistics(self, stat):
        if isinstance(stat, Statistics):
            self.statistics.append(stat)
        else:
            raise TypeError("Must be a Statistics object")

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
        if not compmethods: compmethods = [comparator.HISTCMP_BHATTACHARYYA]
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

            ex.set_feature_matcher(comparator.HistogramsCompare(method, weights))
            self.executions.append(ex)
        print("%d executions generated" % len(self.executions))

    def run(self):
        def order_cols(cols):
            """
            Format of columns to save in Dataframe
            """
            # cols = list(self.dataframe.columns.values)
            ordered_cols = ["Probe", "Gallery", "Preproc", "Segmenter"]
            ordered_cols.extend(sorted([col for col in cols if "Seg" in col and col != "Segmenter"]))
            ordered_cols.extend(["Mask", "Feature_Extractor"])
            ordered_cols.extend(sorted([col for col in cols if "Fe" in col and col != "Feature_Extractor"]))
            ordered_cols.extend(["Regions", "Comparator"])
            ordered_cols.extend(sorted([col for col in cols if "Comp" in col and col != "Comparator"]))
            rest = sorted([item for item in cols if item not in ordered_cols])
            ordered_cols.extend(rest)
            return ordered_cols

        self.dataframe = None
        self._check_initialization()
        total = len(self.executions)
        # time.sleep(5)
        for val, (execution, statistic) in enumerate(zip(self.executions, self.statistics)):
            print("******** Execution %d of %d ********" % (val + 1, total))
            execution.run()
            statistic.set_execution(execution)
            print("Calculating Statistics")
            statistic.run()
            name = statistic.dict_name()
            if self.dataframe is None:
                self.dataframe = pd.DataFrame(name, columns=order_cols(list(name.keys())), index=[0])
            else:
                self.dataframe = self.dataframe.append(name, ignore_index=True)
            print(statistic.rangeX[0])
            print("")  # New lineTL

            #Do some clean up
            execution.unload()
            execution = None
            statistic._exec = None

            gc.collect()

    def _check_initialization(self):
        if not self.executions:
            raise InitializationError
        if self.statistics:
            if len(self.executions) != len(self.statistics):
                if len(self.statistics) != 1:
                    raise InitializationError
                else:
                    for i in range(1, len(self.executions)):
                        self.statistics.append(copy.copy(self.statistics[0]))

    def to_csv(self, path):
        self.dataframe.to_csv(path_or_buf=path, index=False, sep="\t", float_format='%.3f')
        # pd.DataFrame.to_csv(float_format="")

    def to_excel(self, path):
        if os.path.isfile(path):
            df = pd.read_excel(path)
            df = pd.concat([df, self.dataframe])
            # df = df.merge(self.dataframe)
            df.to_excel(excel_writer=path, index=False, float_format='%.3f')
        else:
            self.dataframe.to_excel(excel_writer=path, index=False, float_format='%.3f')
        # pd.DataFrame.to_excel(float_format="")


