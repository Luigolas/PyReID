import itertools
import cv2
from package import image
from package.dataset import Dataset
from package.image import CS_YCrCb, CS_HSV, CS_BGR

__author__ = 'luigolas'

from package.utilities import InitializationError
from package import feature_extractor
import numpy as np


class Preprocessing(object):
    def preprocess(self, dataset):
        raise NotImplementedError("Please Implement preprocess method")

    def convert_image(self, im):
        raise NotImplementedError("Please Implement convert_image method")

    def dict_name(self):
        return {"Preproc": self._method}


class BTF(Preprocessing):
    def __init__(self, method):
        if not (method == "CBTF" or method == "ngMBTF"):  # or method == "gMBTF"
            raise InitializationError("Method " + method + " is not a valid preprocessing method")
        self._method = method
        self.btf = None

    def dict_name(self):
        return {"Preproc": self._method}

    def preprocess(self, dataset):
        self.btf = None
        # if dataset.train_size is None:
        #     raise InitializationError("Can't preprocess if not train/test split defined")
        probe_images = dataset.probe.images_train
        probe_masks = dataset.probe.masks_train
        gallery_images = dataset.gallery.images_train
        gallery_masks = dataset.gallery.masks_train
        if self._method == "CBTF":
            self.btf = self._calc_btf(probe_images, gallery_images, probe_masks, gallery_masks)

        # TODO Consider gMBTF
        # elif self._method == "gMBTF":
        # elements_left = dataset.train_indexes.copy()
        #     btfs = [np.array([0] * 256), np.array([0] * 256), np.array([0] * 256)]
        #     count_btfs = 0
        #     while dataset_len(elements_left) > 0:
        #         individual = [elements_left.pop(0)]
        #         aux_list = []
        #         for elem in elements_left:
        #             if dataset.same_individual(self.probe.files[individual[0]], self.probe.files[elem]):
        #                 individual.append(elem)
        #             else:
        #                 aux_list.append(elem)
        #         elements_left = aux_list
        #         to_compare = [self.gallery.files.index(x) for x in self.gallery.files
        #                       if self.same_individual(self.gallery.files[individual[0]], x)]
        #         # Load images
        #         individual_images = [self.probe.images[x] for x in individual]
        #         to_compare_images = [self.gallery.images[x] for x in to_compare]
        #         masks1 = None
        #         masks2 = None
        #         if self.probe.masks is not None:
        #             masks1 = [self.probe.masks[x] for x in individual]
        #         if self.probe.masks is not None:
        #             masks2 = [self.gallery.masks[x] for x in to_compare]
        #         result = btf(individual_images, to_compare_images, masks1, masks2)
        #         count_btfs += 1
        #         for channel, elem in enumerate(result):
        #             btfs[channel] += elem
        #     f = [np.asarray(np.rint(x / count_btfs), np.int) for x in btfs]

        elif self._method == "ngMBTF":
            btfs = [np.array([0] * 256), np.array([0] * 256), np.array([0] * 256)]  # TODO Generalize
            count_btfs = 0
            for im, mask1 in zip(probe_images, probe_masks):
                # Keep index of images to compare to
                to_compare = [[img, mask] for img, mask in zip(gallery_images, gallery_masks)
                              if dataset.same_individual(im.imgname, img.imgname)]
                for im2, mask2 in to_compare:
                    # btfs.append(btf(im, im2, mask1, mask2))
                    result = self._calc_btf(im, im2, mask1, mask2)
                    count_btfs += 1
                    # for channel, elem in enumerate(result):
                    #     btfs[channel] += elem
                    btfs += result
            self.btf = np.asarray([np.rint(x / count_btfs) for x in btfs], np.int)

        else:
            raise AttributeError("Not a valid preprocessing key")

        if self.btf is None:
            raise NotImplementedError

        new_images_train = []
        for im in dataset.probe.images_train:
            new_images_train.append(self.convert_image(im))
        dataset.probe.images_train = new_images_train

        new_images_test = []
        for im in dataset.probe.images_test:
            new_images_test.append(self.convert_image(im))
        dataset.probe.images_test = new_images_test
        # return new_images

    @staticmethod
    def _calc_btf(im1, im2, mask1, mask2):
        def find_nearest(array, value):
            return (np.abs(array - value)).argmin()

        cumh1 = BTF._cummhist(im1, masks=mask1)
        cumh2 = BTF._cummhist(im2, masks=mask2)
        # For each value in cumh1, look for the closest one (floor, ceil, round?) in cum2, and save index of cum2.
        # func = [np.empty_like(h, np.uint8) for h in cumh1]
        func = np.empty_like(cumh1, np.uint8)
        for f_i, hist_i, hist2_i in zip(func, cumh1, cumh2):  # For each channel
            for index, value in enumerate(hist_i):
                f_i[index] = find_nearest(hist2_i, value)
        return func

    @staticmethod
    def _cummhist(ims, colorspace=image.CS_BGR, masks=None):
        ranges = feature_extractor.Histogram.color_ranges[colorspace]
        bins = [int(b - a) + 1 for a, b in zip(ranges, ranges[1:])[::2]]  # http://stackoverflow.com/a/5394908/3337586
        ev = feature_extractor.Histogram(colorspace, bins=bins, dimension="1D")

        if type(ims) is image.Image:
            ims = [ims]
        if type(masks) is not list:
            masks = [masks] * len(ims)
        h = []
        for im, mask in zip(ims, masks):
            result = ev.transform(im, mask, normalization=None)
            h = [a + b for a, b in
                 itertools.izip_longest(h, list(result), fillvalue=0)]  # Accumulate with previous histograms

        # Normalize each histogram
        num_channels = len(feature_extractor.Histogram.color_channels[colorspace])
        h = np.asarray(h).reshape(num_channels, len(h) / num_channels)
        return np.asarray(
            [feature_extractor.Histogram.normalize_hist(h_channel.cumsum(), normalization=cv2.NORM_INF) for h_channel
             in h])

    def convert_image(self, im):
        im_converted = np.empty_like(im)
        for row in range(im.shape[0]):
            for column in range(im.shape[1]):
                pixel = im[row, column]
                for channel, elem in enumerate(pixel):
                    im_converted[row, column, channel] = self.btf[channel][elem]
        imgname = im.imgname.split(".")
        imgname = ".".join(imgname[:-1]) + self._method + "." + imgname[-1]
        return image.Image(im_converted, colorspace=im.colorspace, imgname=imgname)


class Illumination_Normalization(Preprocessing):
    def __init__(self, color_space=CS_YCrCb):
        self.color_space = color_space
        if color_space == CS_HSV:
            self.channel = 2
        else:  # CS_YCrCb
            self.channel = 0

    def preprocess(self, dataset):
        assert(type(dataset) == Dataset)
        for index, im in enumerate(dataset.probe.images_test):
            dataset.probe.images_test[index] = self.convert_image(im)

        for index, im in enumerate(dataset.probe.images_train):
            dataset.probe.images_train[index] = self.convert_image(im)

        for index, im in enumerate(dataset.gallery.images_test):
            dataset.gallery.images_test[index] = self.convert_image(im)

        for index, im in enumerate(dataset.gallery.images_train):
            dataset.gallery.images_train[index] = self.convert_image(im)

    def convert_image(self, im):
        origin_color_space = im.colorspace
        im = im.to_color_space(self.color_space)
        im[:, :, self.channel] = cv2.equalizeHist(im[:, :, 0])
        return im.to_color_space(origin_color_space)

    def dict_name(self):
        if self.color_space == CS_HSV:
            colorspace = "HSV"
        else:  # CS_YCrCb
            colorspace = "YCrCb"
        return {"Preproc": "IluNorm_%s" % colorspace}