from sklearn.cross_validation import ShuffleSplit, _validate_shuffle_split

__author__ = 'luigolas'


from package.image_set import ImageSet
import re
import numpy as np
# import package.image as image
# import package.feature_extractor as feature_extractor
# import itertools
# import cv2


class Dataset(object):
    def __init__(self, probe=None, gallery=None):
        if probe is not None:
            if "viper" in probe:
                self.id_regex = "[0-9]{3}_"
            elif "CUHK" in probe:
                self.id_regex = "P[1-6]_[0-9]{3}"
            else:  # Default to viper name convection
                self.id_regex = "[0-9]{3}_"

        if probe is not None:
            self.probe = ImageSet(probe)
        else:
            self.probe = None
        if gallery is not None:
            self.gallery = ImageSet(gallery)
        else:
            self.gallery = None
        self.train_indexes = None
        self.test_indexes = None
        self.preprocessed_probe = None

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
        if self.train_indexes is not None:
            name.update({"Train": len(self.train_indexes), "Test": len(self.test_indexes)})

        return name

    def same_individual(self, probe_name, gallery_name):
        elem_id1 = re.search(self.id_regex, probe_name).group(0)
        elem_id2 = re.search(self.id_regex, gallery_name).group(0)
        return elem_id1 == elem_id2

    def same_individual_by_id(self, probe_id, gallery_id, set=None):
        if set == "train":
            probe_name = self.probe.files[self.train_indexes[probe_id]]
            gallery_name = self.gallery.files[self.train_indexes[gallery_id]]
        if set == "test":
            probe_name = self.probe.files[self.test_indexes[probe_id]]
            gallery_name = self.gallery.files[self.test_indexes[gallery_id]]
        elif set is None:
            probe_name = self.probe.files[probe_id]
            gallery_name = self.gallery.files[gallery_id]
        else:
            raise ValueError("set must be None, \"train\" or \"test\"")
        return self.same_individual(probe_name, gallery_name)

    def calc_masks(self, segmenter):
        self.probe.calc_masks(segmenter)
        self.gallery.calc_masks(segmenter)

    def generate_train_set(self, train_size=20, test_size=None):
        """

        :param train_size: float or int (default=20)
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples.
        :return:
        """
        total_len = self.probe.len  # Assumes same gallery and probe size
        train_size, test_size = _validate_shuffle_split(total_len, test_size, train_size)  # Makes sure values are valid
        generator = np.random.RandomState()
        reordered = generator.permutation(total_len)
        self.train_indexes = reordered[:train_size]  # Take the first ones
        self.test_indexes = reordered[-test_size:]  # Take the last ones

    def unload(self):
        self.probe.images = None
        self.probe.masks = None
        self.probe.files = None
        self.gallery.images = None
        self.gallery.masks = None
        self.gallery.files = None
        self.preprocessed_probe = None
        self.train_indexes = None
        self.test_indexes = None
        del self.gallery
        del self.probe
        del self.preprocessed_probe
