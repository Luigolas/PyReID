__author__ = 'luigolas'


from package.image_set import ImageSet
import re
# import numpy as np
# import package.image as image
# import package.evaluator as evaluator
# import itertools
# import cv2


class Dataset():
    def __init__(self):
        self.id_regex = "P[1-6]_[0-9]{3}"
        self.probe = None
        self.gallery = None

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

        return name

    def same_individual(self, id1, id2):
        elem_id1 = re.search(self.id_regex, id1).group(0)
        elem_id2 = re.search(self.id_regex, id2).group(0)
        return elem_id1 == elem_id2

    def calc_masks(self, segmenter):
        self.probe.calc_masks(segmenter)
        self.gallery.calc_masks(segmenter)

    def unload(self):
        self.probe.images = None
        self.probe.masks = None
        self.probe.files = None
        self.gallery.images = None
        self.gallery.masks = None
        self.gallery.files = None
        del self.gallery
        del self.probe
