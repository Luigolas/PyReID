__author__ = 'luigolas'

from package.image_set import ImageSet
import re
# import numpy as np
from sklearn.cross_validation import train_test_split


class Dataset(object):
    def __init__(self, probe=None, gallery=None, train_size=None, test_size=None):
        self.probe = ImageSet(probe)
        self.gallery = ImageSet(gallery)
        if "viper" in probe:
            self.id_regex = "[0-9]{3}_"
        elif "CUHK" in probe:
            self.id_regex = "P[1-6]_[0-9]{3}"
        else:  # Default to viper name convection
            self.id_regex = "[0-9]{3}_"

        self.train_size, self.test_size = self.generate_train_set(train_size, test_size)

        # self.train_indexes = None
        # self.test_indexes = None
        # self.preprocessed_probe = None

    def set_probe(self, folder):
        self.probe = ImageSet(folder)

    def set_gallery(self, folder):
        self.gallery = ImageSet(folder)

    def set_id_regex(self, regex):
        self.id_regex = regex

    def name(self):
        """
        example:
            P2_cam1_P2_cam2_Grabcut2OptimalMask_Histogram_IIP_[5, 5, 5]_6R_3D_BHATT
        :return:
        """
        name = "%s_%s" % (self.probe.name, self.gallery.name)

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
        elem_id1 = re.search(self.id_regex, probe_name).group(0)
        elem_id2 = re.search(self.id_regex, gallery_name).group(0)
        return elem_id1 == elem_id2

    def same_individual_by_id(self, probe_id, gallery_id, set=None):
        if set == "train":
            probe_name = self.probe.files_train[probe_id]
            gallery_name = self.gallery.files_train[gallery_id]
        elif set == "test":
            probe_name = self.probe.files_test[probe_id]
            gallery_name = self.gallery.files_test[gallery_id]
        elif set is None:
            probe_name = self.probe.files[probe_id]
            gallery_name = self.gallery.files[gallery_id]
        else:
            raise ValueError("set must be None, \"train\" or \"test\"")
        return self.same_individual(probe_name, gallery_name)

    def generate_train_set(self, train_size=None, test_size=None):
        """

        :param train_size: float or int (default=20)
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples.
        :return:
        """
        if train_size is None and test_size is None:
            pass  # TODO If no train nor test set... Consider what to do

        self.probe.files_train, self.probe.files_test, self.gallery.files_train, self.gallery.files_test = \
            train_test_split(self.probe.files, self.gallery.files, train_size=train_size, test_size=test_size)
        return len(self.probe.files_train), len(self.probe.files_test)
        # TODO: Assuming same gallery and probe size

    def unload(self):
        self.probe.unload()
        self.gallery.unload()
        del self.gallery
        del self.probe

    def load_images(self):
        self.probe.load_images()
        self.gallery.load_images()
