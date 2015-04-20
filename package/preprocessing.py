import itertools
import cv2
from package import image
from package.dataset import Dataset
from package.image import Image, CS_YCrCb, CS_HSV, CS_BGR
from sklearn.externals.joblib import Parallel, delayed
import numpy as np
from package.utilities import InitializationError
from package import feature_extractor


__author__ = 'luigolas'


class Preprocessing(object):
    def preprocess(self, dataset, n_jobs=1):
        raise NotImplementedError("Please Implement preprocess method")

    def dict_name(self):
        raise NotImplementedError("Please Implement dict_name")


class BTF(Preprocessing):
    def __init__(self, method):
        if not (method == "CBTF" or method == "ngMBTF"):  # or method == "gMBTF"
            raise InitializationError("Method " + method + " is not a valid preprocessing method")
        self._method = method
        self.btf = None

    def dict_name(self):
        return {"Preproc": self._method}

    def preprocess(self, dataset, n_jobs=1):
        print("   BTF (%s)..." % self._method)
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
            new_images_train.append(self._transform_image(im))
        dataset.probe.images_train = new_images_train

        new_images_test = []
        for im in dataset.probe.images_test:
            new_images_test.append(self._transform_image(im))
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

    def _transform_image(self, im):
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

    def preprocess(self, dataset, n_jobs=1):
        print("   Illumination Normalization...")
        assert(type(dataset) == Dataset)
        for index, im in enumerate(dataset.probe.images_test):
            dataset.probe.images_test[index] = self._transform_image(im)

        for index, im in enumerate(dataset.probe.images_train):
            dataset.probe.images_train[index] = self._transform_image(im)

        for index, im in enumerate(dataset.gallery.images_test):
            dataset.gallery.images_test[index] = self._transform_image(im)

        for index, im in enumerate(dataset.gallery.images_train):
            dataset.gallery.images_train[index] = self._transform_image(im)

    def _transform_image(self, im):
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


class Grabcut(Preprocessing):
    compatible_color_spaces = [CS_BGR]

    def __init__(self, mask_source, iter_count=2, color_space=CS_BGR):
        self._mask_name = mask_source.split("/")[-1].split(".")[0]
        self._mask = np.loadtxt(mask_source, np.uint8)
        self._iter_count = iter_count
        if color_space not in Grabcut.compatible_color_spaces:
            raise AttributeError("Grabcut can't work with colorspace " + str(color_space))
        self._colorspace = color_space
        self.name = type(self).__name__ + str(self._iter_count) + self._mask_name
        # self.dict_name = {"Segmenter": str(type(self).__name__), "SegIter": self._iter_count,
        #                   "Mask": self._mask_name}

    def preprocess(self, dataset, n_jobs=1):
        print("   Generating Masks (Grabcut)...")
        imgs = dataset.probe.images_train + dataset.probe.images_test
        imgs += dataset.gallery.images_train + dataset.gallery.images_test
        results = Parallel(n_jobs)(delayed(_parallel_calc_masks)(self.segment, im) for im in imgs)
        train_len = dataset.train_size
        test_len = dataset.test_size
        dataset.probe.masks_train = results[:train_len]
        dataset.probe.masks_test = results[train_len:train_len + test_len]
        dataset.gallery.masks_train = results[train_len + test_len:-test_len]
        dataset.gallery.masks_test = results[-test_len:]

    def segment(self, image):
        """

        :param image:
        :return: :raise TypeError:
        """
        if not isinstance(image, Image):
            raise TypeError("Must be a valid Image (package.image) object")

        if image.colorspace != self._colorspace:
            raise AttributeError("Image must be in BGR color space")

        # if app.DB:
        #     try:
        #         mask = app.DB[self.dbname(image.imgname)]
        #         # print("returning mask for " + imgname + " [0][0:5]: " + str(mask[4][10:25]))
        #         return mask
        #     except FileNotFoundError:
        #         # Not in DataBase, continue calculating
        #         pass

        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        mask = self._mask.copy()
        # mask = copy.copy(self._mask)
        cv2.grabCut(image, mask, None, bgdmodel, fgdmodel, self._iter_count, cv2.GC_INIT_WITH_MASK)

        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # if app.DB:
        #     app.DB[self.dbname(image.imgname)] = mask

        return mask

    def dict_name(self):
        return {"Segmenter": str(type(self).__name__), "SegIter": self._iter_count,
                "Mask": self._mask_name}

    def dbname(self, imgname):
        class_name = type(self).__name__
        foldername = imgname.split("/")[-2]
        imgname = imgname.split("/")[-1]
        imgname = imgname.split(".")[0]  # Take out file extension
        keys = ["masks", class_name, "iter" + str(self._iter_count), self._mask_name, foldername, imgname]
        return keys


def _parallel_calc_masks(seg, im):
    return seg(im)
    # return seg.segment(im)