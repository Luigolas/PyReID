import copy
import package.app as app
import cv2
import numpy as np
from package.image import Image, CS_BGR

__author__ = 'luigolas'


class Segmenter():
    compatible_color_spaces = []

    def segment(self, image):
        """

        :param image:
        :raise NotImplementedError:
        """
        raise NotImplementedError("Please Implement segment method")


class Grabcut(Segmenter):
    compatible_color_spaces = [CS_BGR]

    def __init__(self, mask_source, iterCount, color_space=CS_BGR):
        self._mask_name = mask_source.split("/")[-1].split(".")[0]
        self._mask = np.loadtxt(mask_source, np.uint8)
        self._iterCount = iterCount
        if color_space not in Grabcut.compatible_color_spaces:
            raise AttributeError("Grabcut can't work with colorspace " + str(color_space))
        self._colorspace = color_space
        self.name = type(self).__name__ + str(self._iterCount) + self._mask_name
        self.dict_name = {"Segmenter": str(type(self).__name__), "SegIter": self._iterCount,
                          "Mask": self._mask_name}

    def segment(self, image):
        """

        :param image:
        :return: :raise TypeError:
        """
        if not isinstance(image, Image):
            raise TypeError("Must be a valid Image (package.image) object")

        if image.colorspace != self._colorspace:
            raise AttributeError("Image must be in BGR color space")

        if app.DB:
            try:
                mask = app.DB[self.dbname(image.imgname)]
                # print("returning mask for " + imgname + " [0][0:5]: " + str(mask[4][10:25]))
                return mask
            except FileNotFoundError:
                # Not in DataBase, continue calculating
                pass

        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        # mask = self._mask.copy()
        mask = copy.copy(self._mask)
        cv2.grabCut(image, mask, None, bgdmodel, fgdmodel, self._iterCount, cv2.GC_INIT_WITH_MASK)

        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        if app.DB:
            app.DB[self.dbname(image.imgname)] = mask

        return mask

    def dbname(self, imgname):
        classname = type(self).__name__
        foldername = imgname.split("/")[-2]
        imgname = imgname.split("/")[-1]
        imgname = imgname.split(".")[0]  # Take out file extension
        keys = ["masks", classname, "iter" + str(self._iterCount), self._mask_name, foldername, imgname]
        return keys
