__author__ = 'luigolas'

from package.image_set import ImageSet
import re
from sklearn.cross_validation import ShuffleSplit
from itertools import chain
from sklearn.utils import safe_indexing
import numpy as np

class Dataset(object):
    """

    :param probe:
    :param gallery:
    :param train_size:
    :param test_size:
    """

    def __init__(self, probe=None, gallery=None, train_size=None, test_size=None):
        self.probe = ImageSet(probe)
        self.gallery = ImageSet(gallery)
        if "viper" in probe:
            self.id_regex = "[0-9]{3}_"
        elif "CUHK" in probe:
            self.id_regex = "P[1-6]_[0-9]{3}"
        else:  # Default to viper name convection
            self.id_regex = "[0-9]{3}_"
        self.train_indexes, self.test_indexes = [], []
        self.train_size = 0
        self.test_size = 0
        self.generate_train_set(train_size, test_size)

    def set_probe(self, folder):
        """

        :param folder:
        :return:
        """
        self.probe = ImageSet(folder)

    def set_gallery(self, folder):
        """

        :param folder:
        :return:
        """
        self.gallery = ImageSet(folder)

    def set_id_regex(self, regex):
        """

        :param regex:
        :return:
        """
        self.id_regex = regex

    def name(self):
        """
        example:
            P2_cam1_P2_cam2_Grabcut2OptimalMask_Histogram_IIP_[5, 5, 5]_6R_3D_BHATT
        :return:
        """
        name = "%s_%s_Train%s_Test%s" % (self.probe.name, self.gallery.name, self.train_size, self.test_size)

        return name

    def dict_name(self):
        """
        example:
        name = {"Probe": "P2_cam1", "Gallery": "P2_cam2", "Segment": "Grabcut", "SegIter": "2", "Mask": "OptimalMask",
                "Evaluator": "Histogram", "EvColorSpace": "IIP", "EvBins": "[5, 5, 5]", "EvDims": "3D", "Regions": "6R",
                "Comparator": "BHATT"}
        :return:
        """
        name = {"Probe": self.probe.name, "Gallery": self.gallery.name}
        # if self.train_indexes is not None:
        name.update({"Train": self.train_size, "Test": self.test_size})

        return name

    def same_individual(self, probe_name, gallery_name):
        """

        :param probe_name:
        :param gallery_name:
        :return:
        """
        elem_id1 = re.search(self.id_regex, probe_name).group(0)
        elem_id2 = re.search(self.id_regex, gallery_name).group(0)
        return elem_id1 == elem_id2

    def same_individual_by_pos(self, probe_pos, gallery_pos, selected_set=None):
        """

        :param probe_pos:
        :param gallery_pos:
        :param selected_set:
        :return:
        """
        if selected_set == "train":
            probe_name = self.probe.files_train[probe_pos]
            gallery_name = self.gallery.files_train[gallery_pos]
        elif selected_set == "test":
            probe_name = self.probe.files_test[probe_pos]
            gallery_name = self.gallery.files_test[gallery_pos]
        elif selected_set is None:
            probe_name = self.probe.files[probe_pos]
            gallery_name = self.gallery.files[gallery_pos]
        else:
            raise ValueError("selected_set must be None, \"train\" or \"test\"")
        return self.same_individual(probe_name, gallery_name)

    def generate_train_set(self, train_size=None, test_size=None, rand_state=None):
        """



        :param test_size:
        :param rand_state:
        :param train_size: float or int (default=20)
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples.
        :return:
        """
        # self.probe.clear()
        # self.gallery.clear()

        if train_size is None and test_size is None:
            self.probe.files_train, self.probe.files_test = [], self.probe.files
            self.gallery.files_train, self.gallery.files_test = [], self.gallery.files
            self.train_indexes, self.test_indexes = [], list(range(0, len(self.probe.files)))
        else:
            n_samples = len(self.probe.files)
            cv = ShuffleSplit(n_samples, test_size=test_size, train_size=train_size, random_state=rand_state)
            train_indexes, test_indexes = next(iter(cv))
            arrays = [self.probe.files, self.gallery.files]
            self.probe.files_train, self.probe.files_test, self.gallery.files_train, self.gallery.files_test = \
                list(chain.from_iterable((safe_indexing(a, train_indexes),
                                          safe_indexing(a, test_indexes)) for a in arrays))
            self.train_indexes, self.test_indexes = train_indexes, test_indexes

        self.train_size = len(self.train_indexes)
        self.test_size = len(self.test_indexes)
        # self.probe.files_train, self.probe.files_test, self.gallery.files_train, self.gallery.files_test = \
        #     train_test_split(self.probe.files, self.gallery.files, train_size=train_size, test_size=test_size,
        #                      random_state=0)
        # TODO: Assuming same gallery and probe size

    def change_probe_and_gallery(self, probe_list, gallery_list, train_size=0):
        """

        :param probe_list:
        :param gallery_list:
        :return:
        """
        probe_files = [(idx, f) for idx, f in enumerate(self.probe.files) for pl_elem in probe_list if pl_elem in f]
        self.probe.files_test = [elem[1] for elem in probe_files]
        self.test_indexes = [elem[0] for elem in probe_files]
        self.gallery.files_test = [f for f in self.gallery.files for gl_elem in gallery_list if gl_elem in f]
        self.test_size = len(self.test_indexes)
        if train_size > 0:
            self.train_size = train_size
            # http://stackoverflow.com/a/15940459/3337586
            mask = np.in1d(self.probe.files, self.probe.files_test)
            probe_train_files = [self.probe.files[i] for i in np.where(~mask)[0]]
            permutation = np.random.RandomState().permutation(len(probe_train_files))
            ind_test = permutation[:train_size]
            self.train_indexes = ind_test
            self.probe.files_train = [probe_train_files[i] for i in ind_test]
            gallery_train_files = [self.gallery.files[i] for i in np.where(~mask)[0]]
            self.gallery.files_train = [gallery_train_files[i] for i in ind_test]

    def load_images(self):
        """

        :return:
        """
        self.probe.load_images()
        self.gallery.load_images()

    # def unload(self):
    #     self.probe.unload()
    #     self.gallery.unload()
    #     del self.gallery
    #     del self.probe
