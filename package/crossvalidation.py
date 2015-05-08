__author__ = 'luigolas'

import os
from package.statistics import Statistics
import pandas as pd
import numpy as np
import random
import pickle
import shelve


class CrossValidation():
    """

    :param execution:
    :param splits_file: If passed, read splittings from file and num_validations, and test_size is ignored
    :param num_validations:
    :param train_size:
    :param test_size:
    :return:
    """
    def __init__(self, execution, splits_file=None, num_validations=10, train_size=0, test_size=0.5):
        self.execution = execution
        self.statistics = []
        self.mean_stat = Statistics()

        if splits_file:
            self.files, self.num_validations, self.test_size = self._read_split_file(splits_file)
            self.execution.dataset.change_probe_and_gallery(self.files[0][0], self.files[0][1],
                                                            train_size=train_size)
        else:
            self.num_validations = num_validations
            self.test_size = test_size
            self.files = None
            self.execution.dataset.generate_train_set(train_size=train_size, test_size=self.test_size)

        self.train_size = self.execution.dataset.train_size
        self.dataframe = None

    @staticmethod
    def _read_split_file(path):
        """
        File format:
        % Experiment Number 1

        1 - cam_a/001_45.bmp  cam_b/001_90.bmp
        2 - cam_a/002_45.bmp  cam_b/002_90.bmp
        3 - cam_a/003_0.bmp  cam_b/003_90.bmp
        [...]
        316 - cam_a/872_0.bmp  cam_b/872_180.bmp


        % Experiment Number 2

        1 - cam_a/000_45.bmp  cam_b/000_45.bmp
        2 - cam_a/001_45.bmp  cam_b/001_90.bmp
        ....


        It defines only tests elements.

        :param execution:
        :param path:
        :return:
        """
        sets = []
        with open(path, 'r') as f:
            new_set = [[], []]
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if "% " in line:
                    if new_set[0]:
                        sets.append(new_set)
                        new_set = [[], []]
                    continue

                line = line.split(" - ")[-1]
                line = line.split("  ")
                new_set[0].append(line[0])
                new_set[1].append(line[1])
        sets.append(new_set)

        return sets, len(sets), len(sets[0][0])

    def run(self, verbosity=2):
        """

        :return:
        """
        self.dataframe = None
        # self._check_initialization()

        if self.train_size == 0 and not self.files:
            self.execution.dataset.generate_train_set(train_size=None, test_size=None)
            if verbosity >= 0: print("******** Execution 1 of 1 ********")
            r_m = self.execution.run(verbosity)
            if verbosity > 0: print("Calculating Statistics")
            for val in range(self.num_validations):
                self.execution.dataset.generate_train_set(train_size=0, test_size=self.test_size)
                statistic = Statistics()
                statistic.run(self.execution.dataset, r_m)
                self.statistics.append(statistic)
        else:
            for val in range(self.num_validations):
                if verbosity >= 0: print("******** Execution %d of %d ********" % (val + 1, self.num_validations))
                if self.files:
                    self.execution.dataset.change_probe_and_gallery(self.files[val][0], self.files[val][1],
                                                                    train_size=self.train_size)
                else:
                    self.execution.dataset.generate_train_set(train_size=self.train_size, test_size=self.test_size)
                r_m = self.execution.run(verbosity)
                if verbosity > 0: print("Calculating Statistics")
                statistic = Statistics()
                statistic.run(self.execution.dataset, r_m)
                self.statistics.append(statistic)

        CMCs = np.asarray([stat.CMC for stat in self.statistics])
        self.mean_stat.CMC = np.sum(CMCs, axis=0) / float(self.num_validations)
        self.mean_stat.AUC = np.mean([stat.AUC for stat in self.statistics])
        mean_values = np.asarray([stat.mean_value for stat in self.statistics])
        self.mean_stat.mean_value = np.mean(mean_values)
        if verbosity > 2:
            print "Range 20: %f" % self.mean_stat.CMC[19]
            print "AUC: %f" % self.mean_stat.AUC

    def dict_name(self, use_stats=True):
        """


        :param use_stats:
        :return:
        """
        name = {}
        name.update(self.execution.dict_name())
        if use_stats:
            name.update(self.mean_stat.dict_name())
        name.update({"NumValidations": self.num_validations})
        return name

    # def to_csv(self, path):
    #     self.dataframe.to_csv(path_or_buf=path, index=False, sep="\t", float_format='%.3f')
    #     # pd.DataFrame.to_csv(float_format="")

    def id(self):
        to_encode = str(self.dict_name(False)) + str([type(i).__name__ for i in self.execution.preprocessing])
        return str(abs(hash(to_encode)))

    def id_is_saved(self, db_file, id=None):
        if not id:
            id = self.id()

        db = shelve.open(db_file, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            val = db.has_key(str(id))
        finally:
            db.close()
        return val

    def to_excel(self, path, with_id=False, sorting=False):
        """

        :param path:
        :param with_id:
        :param sorting:
        :return: id
        """
        data = self.dict_name()
        dataframe = pd.DataFrame(data, columns=self.order_cols(list(data.keys())), index=[0])
        if with_id:
            # data_id = random.randint(0, 1000000000)
            data_id = self.id()
            dataframe['id'] = data_id
        else:
            data_id = None

        if os.path.isfile(path):
            df = pd.read_excel(path)
            df = pd.concat([df, dataframe])
            # df = df.merge(self.dataframe)
            if sorting:
                cols = sorted([col for col in list(df.columns.values) if "Range" in col], reverse=True)
                df.sort(columns=cols, ascending=False, inplace=True)

            df.to_excel(excel_writer=path, index=False, columns=self.order_cols(list(df.keys())), float_format='%.3f')
        else:
            dataframe.to_excel(excel_writer=path, index=False, float_format='%.3f')

        return data_id

    def save_stats(self, db_file, data_id):
        """

        :param db_file:
        :param data_id:
        :return:
        """

        db = shelve.open(db_file, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            data = {"CMC": self.mean_stat.CMC, "AUC": self.mean_stat.AUC, "mean_value": self.mean_stat.mean_value,
                    "name": self.dict_name()}
            db[str(data_id)] = data
        finally:
            db.close()
        return data

    @staticmethod
    def load_stats(db_file, data_id):
        """

        :param db_file:
        :param data_id:
        :return:
        """
        db = shelve.open(db_file, protocol=pickle.HIGHEST_PROTOCOL, flag='r')
        try:
            data = db[str(data_id)]
        except KeyError:
            data = None
        finally:
            db.close()

        return data

    @staticmethod
    def order_cols(cols):
        """
        Format of columns to save in Dataframe
        :param cols:
        """
        ordered_cols = ["Probe", "Gallery", "Train", "Test"]
        ordered_cols.extend(sorted([col for col in cols if "Preproc" in col]))

        # ordered_cols = ["Probe", "Gallery", "Train", "Test", "Segmenter"]
        # ordered_cols.extend(sorted([col for col in cols if "Seg" in col and col != "Segmenter"]))
        # ordered_cols.extend(["Normalization"])
        # ordered_cols.extend(sorted([col for col in cols if "Norm" in col and col != "Normalization"]))
        # ordered_cols.extend(["BTF", "Regions"])
        # ordered_cols.extend(sorted([col for col in cols if "Reg" in col and col != "Regions"]))
        # ordered_cols.extend(["Map"])
        # ordered_cols.extend(sorted([col for col in cols if "Map" in col and col != "Map"]))

        ordered_cols.extend(["Feature_Extractor"])
        ordered_cols.extend(sorted([col for col in cols if "Fe" in col and col != "Feature_Extractor"]))
        ordered_cols.extend(["FMatcher"])
        ordered_cols.extend(sorted([col for col in cols if "FM" in col and col != "FMatcher"]))
        ordered_cols.extend(sorted([col for col in cols if "Range" in col]))
        ordered_cols.extend(["AUC", "MeanValue", "NumValidations"])
        rest = sorted([item for item in cols if item not in ordered_cols])
        ordered_cols.extend(rest)
        return ordered_cols


    # def add_execs_grabcut_btf_histograms(self, probe, gallery, id_regex=None, segmenter_iter=None, masks=None,
    #                                  colorspaces=None, binss=None, regions=None, region_name=None, weights=None,
    #                                  dimensions=None, compmethods=None, preprocessing=None, train_test_split=None):
    #     """
    #     :param probe:
    #     :param gallery:
    #     :param id_regex:
    #     :param segmenter_iter:
    #     :param masks:
    #     :param colorspaces:
    #     :param binss: can be a list of bins or a tuple, like in range. ex:
    #                   binss = [[40,40,40], [50,50,50]]
    #                   binss = (40, 51, 10) # Will generate same result
    #     :param regions:
    #     :param dimensions:
    #     :param compmethods:
    #     :return:
    #     """
    #     print("Generating executions...")
    #     if not segmenter_iter: segmenter_iter = [2]
    #     if not compmethods: compmethods = [feature_matcher.HISTCMP_BHATTACHARYYA]
    #     if not dimensions: dimensions = ["1D"]
    #     if not colorspaces: colorspaces = [CS_BGR]
    #     if not masks: raise InitializationError("Mask needed")
    #     # if not regions: regions = [None]
    #     if not preprocessing:
    #         preprocessing = [None]
    #     if not binss:
    #         binss = [[32, 32, 32]]
    #     elif isinstance(binss, tuple):
    #         binss_temp = []
    #         for r in range(*binss):
    #             binss_temp.append([r, r, r])
    #         binss = binss_temp
    #     if train_test_split is None:
    #         train_test_split = [[10, None]]
    #     # elif type(train_test_split) != list:
    #     #     train_test_split = [train_test_split]
    #
    #     for gc_iter, mask, colorspace, bins, dimension, method, preproc, split in itertools.product(
    #             segmenter_iter, masks, colorspaces, binss, dimensions, compmethods, preprocessing, train_test_split):
    #
    #         if preproc is None:
    #             btf = None
    #         else:
    #             btf = BTF(preproc)
    #         ex = Execution(Dataset(probe, gallery, split[0], split[1]), Grabcut(mask, gc_iter, CS_BGR), btf)
    #
    #         if id_regex:
    #             ex.set_id_regex(id_regex)
    #
    #         # bin = bins[0:len(Histogram.color_channels[colorspace])]
    #         ex.set_feature_extractor(
    #             Histogram(colorspace, bins, regions=regions, dimension=dimension, region_name=region_name))
    #
    #         ex.set_feature_matcher(feature_matcher.HistogramsCompare(method, weights))
    #         self.executions.append(ex)
    #     print("%d executions generated" % len(self.executions))

