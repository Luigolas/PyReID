__author__ = 'luigolas'

import numpy as np
import cv2
from package.utilities import safe_ln, FileNotFoundError
import package.app as app

CS_IIP = 1
CS_BGR = 2
CS_HSV = 3

colorspace_name = ["", "IIP", "BGR", "HSV"]

iipA = np.asarray([[27.07439, -0.2280783, -1.806681],
                   [-5.646736, -7.722125, 12.86503],
                   [-4.163133, -4.579428, -4.576049]])

iipB = np.asarray([[0.9465229, 0.2946927, -0.1313419],
                   [-0.1179179, 0.9929960, 0.007371554],
                   [0.09230461, -0.04645794, 0.9946464]])

# iip_min = -576.559
# iip_min = -622.28125
# iip_max = 306.672
iip_min = -73.670
iip_max = 140.333

bgr2iipClip = 3  # Minimal value for RGB image before converting to IIP

hsvmax = [179, 255, 255]


class Image(np.ndarray):
    """
    Image class, basically an ndarray with imgname and colorspace parameters
    --To make it work with multiprocessing Queue it needs to be pickeable (reduce and getstate)
    ----More info: http://mail.scipy.org/pipermail/numpy-discussion/2007-April/027193.html

    :param imgname:
    :param colorspace:
    :return:
    :raise FileNotFoundError:
    """

    def __new__(cls, ndarray, colorspace=CS_BGR, imgname=None):
        if not isinstance(ndarray, np.ndarray):
            raise ValueError('Argument must be a valid numpy.ndarray')
        return ndarray.view(cls)

    def __init__(self, src, colorspace=CS_BGR, imgname=None):
        if not imgname and isinstance(src, Image):
            imgname = src.imgname
        self.imgname = imgname
        self.colorspace = colorspace
        # This _method should NOT affect DB

    def __deepcopy__(self, memo):
        # Allowing deepcopy of Images
        result = Image(self.copy(), self.colorspace, self.imgname)
        memo[id(self)] = result
        return result

    def __copy__(self):
        # Allowing copy of Images
        result = Image(self.copy(), self.colorspace, self.imgname)
        return result

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.colorspace = getattr(obj, 'colorspace', CS_BGR)

    def __reduce__(self):
        # Enables pickling, needed for multiprocessing
        object_state = list(np.ndarray.__reduce__(self))
        subclass_state = (self.imgname, self.colorspace)
        object_state[2] = (object_state[2], subclass_state)
        return tuple(object_state)

    def __setstate__(self, state):
        # Enables pickling, needed for multiprocessing
        nd_state, own_state = state
        np.ndarray.__setstate__(self, nd_state)

        imgname, colorspace = own_state
        self.imgname, self.colorspace = imgname, colorspace

    @classmethod
    def from_filename(cls, imgname, colorspace=CS_BGR):
        """

        :param imgname:
        :param colorspace:
        :return: :raise FileNotFoundError:
        """
        img = cv2.imread(imgname)
        if img is None:
            raise FileNotFoundError("Image file not found at specified path")
        img = Image(img)
        img.colorspace = CS_BGR
        img.imgname = imgname
        if colorspace != CS_BGR:
            img = img.to_color_space(colorspace)
        return img

    def to_color_space(self, colorspace):
        """

        :param colorspace:
        :return:
        """
        # Attempt to load from DB
        try:
            if app.DB:
                img = app.DB[self.dbname(colorspace)]
                img = Image(img, colorspace)
                img.imgname = self.imgname
                return img
        except FileNotFoundError:
            #Not in DataBase, continue calculating
            pass
        img = None
        if self.colorspace == CS_BGR:
            if colorspace == CS_IIP:
                img = self._bgr2iip()
            elif colorspace == CS_HSV:
                img = self._bgr2hsv()
        img.imgname = self.imgname
        #Save in Connection
        if app.DB:
            app.DB[img.dbname()] = img
        return img

    def dbname(self, colorspace=None):
        if not self.imgname:
            raise AttributeError("Image has no imgname")
        if not colorspace:
            colorspace = self.colorspace
        foldername = self.imgname.split("/")[-2]
        imgname = self.imgname.split("/")[-1]
        imgname = imgname.split(".")[0]
        keys = ["images", str(colorspace), foldername, imgname]
        return keys

    def _bgr2iip(self):

        # Convert to CV_32F3 , floating point in range 0.0 , 1.0
        # imgf32 = np.float32(self)
        # imgf32 = imgf32*1.0/255

        # Clip values to min value
        src = self.clip(bgr2iipClip)

        # src_xyz = cv2.cvtColor(imgf32, cv2.COLOR_BGR2XYZ)
        src_xyz = cv2.cvtColor(src, cv2.COLOR_BGR2XYZ)

        img_iip = np.empty_like(src_xyz, np.float32)

        for row_index, row in enumerate(src_xyz):
            for columm_index, element in enumerate(row):
                element = np.dot(iipB, element)
                element = safe_ln(element)  # element = np.log(element)
                element = np.dot(iipA, element)
                # element = (element - iip_min)/(iip_max - iip_min)  # Normalized to 0.0 ; 1.0
                # element = np.around(element, decimals=6)  # Remove some "extreme" precision
                img_iip[row_index][columm_index] = element
        img_iip = Image(img_iip, CS_IIP)
        return img_iip

    def _bgr2hsv(self):
        img = cv2.cvtColor(self, cv2.COLOR_BGR2HSV)
        img = Image(img, CS_HSV)
        return img
