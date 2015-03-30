import itertools
import cv2
from package import image

__author__ = 'luigolas'

from package.utilities import InitializationError
from package import feature_extractor
import numpy as np


class Preprocessing(object):
    def __init__(self, method):
        if not (method == "CBTF" or method == "ngMBTF"):  # or method == "gMBTF"
            raise InitializationError("Method " + method + " is not a valid preprocessing _method")
        self._method = method

    def preprocess(self, dataset):
        f = None
        # probe_images = np.asarray(dataset.probe.images)[dataset.train_indexes]
        # probe_masks = np.asarray(dataset.probe.masks)[dataset.train_indexes]
        # gallery_images = np.asarray(dataset.gallery.images)[dataset.train_indexes]
        # gallery_masks = np.asarray(dataset.gallery.masks)[dataset.train_indexes]
        if dataset.train_indexes is None:
            raise InitializationError("Can't preprocess if not train/test split defined")
        probe_images = [dataset.probe.images[i] for i in dataset.train_indexes]
        probe_masks = [dataset.probe.masks[i] for i in dataset.train_indexes]
        gallery_images = [dataset.gallery.images[i] for i in dataset.train_indexes]
        gallery_masks = [dataset.gallery.masks[i] for i in dataset.train_indexes]
        if self._method == "CBTF":
            f = self._btf(probe_images, gallery_images, probe_masks, gallery_masks)

        # TODO Consider gMBTF
        # elif self._method == "gMBTF":
        # elements_left = dataset.train_indexes.copy()
        #     btfs = [np.array([0] * 256), np.array([0] * 256), np.array([0] * 256)]
        #     count_btfs = 0
        #     while len(elements_left) > 0:
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
                    result = self._btf(im, im2, mask1, mask2)
                    count_btfs += 1
                    # for channel, elem in enumerate(result):
                    #     btfs[channel] += elem
                    btfs += result
            f = np.asarray([np.rint(x / count_btfs) for x in btfs], np.int)

        else:
            raise AttributeError("Not a valid preprocessing key")

        if f is None:
            raise NotImplementedError



        new_images = []
        for im in dataset.probe.images:
            new_images.append(self.convert_image(f, im, self._method))
        return new_images

    @staticmethod
    def _btf(im1, im2, mask1, mask2):
        def find_nearest(array, value):
            return (np.abs(array - value)).argmin()

        cumh1 = Preprocessing._cummhist(im1, masks=mask1)
        cumh2 = Preprocessing._cummhist(im2, masks=mask2)
        # For each value in cumh1, look for the closest one (floor, ceil, round?) in cum2, and save index of cum2.
        # func = [np.empty_like(h, np.uint8) for h in cumh1]
        func = np.empty_like(cumh1, np.uint8)
        for f_i, hist_i, hist2_i in zip(func, cumh1, cumh2):  # For each channel
            for index, value in enumerate(hist_i):
                f_i[index] = find_nearest(hist2_i, value)
        return func

    @staticmethod
    def _cummhist(ims, colorspace=image.CS_BGR, masks=None):
        ranges = feature_extractor.Histogram.ranges[colorspace]
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
        num_channels = len(feature_extractor.Histogram.channels[colorspace])
        h = np.asarray(h).reshape(num_channels, len(h) / num_channels)
        return np.asarray(
            [feature_extractor.Histogram.normalize_hist(h_channel.cumsum(), normalization=cv2.NORM_INF) for h_channel
             in h])

    @staticmethod
    def convert_image(f, im, method):
        im_converted = np.empty_like(im)
        for row in range(im.shape[0]):
            for column in range(im.shape[1]):
                pixel = im[row, column]
                for channel, elem in enumerate(pixel):
                    im_converted[row, column, channel] = f[channel][elem]
        imgname = im.imgname.split(".")
        imgname = ".".join(imgname[:-1]) + method + "." + imgname[-1]
        return image.Image(im_converted, colorspace=im.colorspace, imgname=imgname)