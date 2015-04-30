import itertools
import cv2
import math
from scipy.optimize import fminbound
from package import image
from package.dataset import Dataset
from package.image import Image, CS_YCrCb, CS_HSV, CS_BGR
from sklearn.externals.joblib import Parallel, delayed
import numpy as np
from package.utilities import InitializationError
from package import feature_extractor
from scipy.stats import norm
from scipy.io.matlab import loadmat


__author__ = 'luigolas'


def _parallel_preprocess(preproc, im, *args):
    return preproc.process_image(im, *args)


class Preprocessing(object):
    def preprocess_dataset(self, dataset, n_jobs=-1):
        raise NotImplementedError("Please Implement preprocess_dataset method")

    # def process_image(self, im, *args):
    #     raise NotImplementedError("Please Implement process_image method")

    def dict_name(self):
        raise NotImplementedError("Please Implement dict_name")


class BTF(Preprocessing):
    def __init__(self, method):
        if not (method == "CBTF" or method == "ngMBTF"):  # or method == "gMBTF"
            raise InitializationError("Method " + method + " is not a valid preprocessing method")
        self._method = method
        self.btf = None

    def dict_name(self):
        return {"BTF": self._method}

    def preprocess_dataset(self, dataset, n_jobs=-1):
        print("   BTF (%s)..." % self._method)
        self.btf = None
        if not dataset.train_size:
            raise InitializationError("Can't preprocess without train elements")
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
            new_images_train.append(self.process_image(im))
        dataset.probe.images_train = new_images_train

        new_images_test = []
        for im in dataset.probe.images_test:
            new_images_test.append(self.process_image(im))
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
            result = ev.extract(im, mask, normalization=None)
            h = [a + b for a, b in
                 itertools.izip_longest(h, list(result), fillvalue=0)]  # Accumulate with previous histograms

        # Normalize each histogram
        num_channels = len(feature_extractor.Histogram.color_channels[colorspace])
        h = np.asarray(h).reshape(num_channels, len(h) / num_channels)
        return np.asarray(
            [feature_extractor.Histogram.normalize_hist(h_channel.cumsum(), normalization=cv2.NORM_INF) for h_channel
             in h])

    def process_image(self, im, *args):
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

    def preprocess_dataset(self, dataset, n_jobs=1):
        print("   Illumination Normalization...")
        assert(type(dataset) == Dataset)

        imgs = dataset.probe.images_train + dataset.probe.images_test
        imgs += dataset.gallery.images_train + dataset.gallery.images_test
        results = Parallel(n_jobs)(delayed(_parallel_preprocess)(self, im) for im in imgs)
        train_len = dataset.train_size
        test_len = dataset.test_size
        dataset.probe.images_train = results[:train_len]
        dataset.probe.images_test = results[train_len:train_len + test_len]
        dataset.gallery.images_train = results[train_len + test_len:-test_len]
        dataset.gallery.images_test = results[-test_len:]

    def process_image(self, im, *args):
        origin_color_space = im.colorspace
        im = im.to_color_space(self.color_space)
        im[:, :, self.channel] = cv2.equalizeHist(im[:, :, self.channel])
        return im.to_color_space(origin_color_space)

    def dict_name(self):
        """

        :return:
        """
        if self.color_space == CS_HSV:
            colorspace = "HSV"
        else:  # CS_YCrCb
            colorspace = "YCrCb"
        return {"Normalization": "IluNorm_%s" % colorspace}


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

    def preprocess_dataset(self, dataset, n_jobs=-1):
        """

        :param dataset:
        :param n_jobs:
        :return:
        """
        print("   Generating Masks (Grabcut)...")
        imgs = dataset.probe.images_train + dataset.probe.images_test
        imgs += dataset.gallery.images_train + dataset.gallery.images_test
        results = Parallel(n_jobs)(delayed(_parallel_preprocess)(self, im) for im in imgs)
        train_len = dataset.train_size
        test_len = dataset.test_size
        dataset.probe.masks_train = results[:train_len]
        dataset.probe.masks_test = results[train_len:train_len + test_len]
        dataset.gallery.masks_train = results[train_len + test_len:-test_len]
        dataset.gallery.masks_test = results[-test_len:]

    def process_image(self, im):
        """

        :param im:
        :return: :raise TypeError:
        """
        if not isinstance(im, Image):
            raise TypeError("Must be a valid Image (package.image) object")

        if im.colorspace != self._colorspace:
            raise AttributeError("Image must be in BGR color space")

        # if app.DB:
        #     try:
        #         mask = app.DB[self.dbname(im.imgname)]
        #         # print("returning mask for " + imgname + " [0][0:5]: " + str(mask[4][10:25]))
        #         return mask
        #     except FileNotFoundError:
        #         # Not in DataBase, continue calculating
        #         pass

        bgdmodel = np.zeros((1, 65), np.float64)
        fgdmodel = np.zeros((1, 65), np.float64)
        mask = self._mask.copy()
        # mask = copy.copy(self._mask)
        cv2.grabCut(im, mask, None, bgdmodel, fgdmodel, self._iter_count, cv2.GC_INIT_WITH_MASK)

        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        # if app.DB:
        #     app.DB[self.dbname(im.imgname)] = mask

        return mask

    def dict_name(self):
        """
        :return: dictionary
        """
        return {"Segmenter": str(type(self).__name__), "SegIter": self._iter_count,
                "SegMask": self._mask_name}

    # def dbname(self, imgname):
    #     class_name = type(self).__name__
    #     foldername = imgname.split("/")[-2]
    #     imgname = imgname.split("/")[-1]
    #     imgname = imgname.split(".")[0]  # Take out file extension
    #     keys = ["masks", class_name, "iter" + str(self._iter_count), self._mask_name, foldername, imgname]
    #     return keys


class MasksFromMat(Preprocessing):
    """

    :param mat_path: path to .mat file to read
    :param var_name: variable name to look for in mat file
    :param invert: If False, first elements are considered as masks for probe. If True, first elements as gallery
    :param pair_order: If True, assumes it's order alternatively
    :return:
    """
    def __init__(self, mat_path, var_name='msk', invert=False, pair_order=True):
        self.pair_order = pair_order
        self.path = mat_path
        self.var_name = var_name
        self.invert = invert
        self.name = None

    def preprocess_dataset(self, dataset, n_jobs=-1):
        """

        :param dataset:
        :param n_jobs:
        :return:
        """
        print("   Loading masks from .mat file")
        data = loadmat(self.path)
        masks = data[self.var_name][0]
        try:
            name = data['msk_name'][0]
            keys = name.dtype.names
            self.name = {}
            for key in keys:
                value = name[0][key]
                if value.dtype == int:
                    value = int(value[0])
                else:
                    value = str(value[0])
                self.name.update({key: value})
        except KeyError:
            self.name = None

        if not self.invert:
            masks_probe = masks.take(range(0, masks.size, 2))
            masks_gallery = masks.take(range(1, masks.size, 2))
        else:
            masks_gallery = masks.take(range(1, masks.size, 2))
            masks_probe = masks.take(range(0, masks.size, 2))

        dataset.probe.masks_train = list(masks_probe[dataset.train_indexes])
        dataset.probe.masks_test = list(masks_probe[dataset.test_indexes])
        dataset.gallery.masks_train = list(masks_gallery[dataset.train_indexes])
        dataset.gallery.masks_test = list(masks_gallery[dataset.test_indexes])

    def dict_name(self):
        """

        :return:
        """
        if self.name is not None:
            return self.name
        else:
            name = self.path.split("/")[-1]
            return {"Segmenter": str(type(self).__name__), "SegMask": name}


class SilhouetteRegionsPartition(Preprocessing):
    def __init__(self, alpha=0.5, sub_divisions=0):
        self.I = 0
        self.J = 0
        self.deltaI = 0
        self.deltaJ = 0
        self.search_range_H = []
        self.alpha = alpha
        self.sub_divisions = int(sub_divisions)

    def preprocess_dataset(self, dataset, n_jobs=-1):
        """

        :param dataset:
        :param n_jobs:
        :return:
        """
        print("   Calculating Silhouette Regions...")
        (self.I, self.J, _) = dataset.probe.images_test[0].shape  # Assumes all images of same shape
        self.deltaI = self.I / 4

        self.search_range_H = [self.deltaI, (self.I - self.deltaI) - 1]

        imgs = dataset.probe.images_train + dataset.probe.images_test
        imgs += dataset.gallery.images_train + dataset.gallery.images_test
        masks = dataset.probe.masks_train + dataset.probe.masks_test
        masks += dataset.gallery.masks_train + dataset.gallery.masks_test

        results = Parallel(n_jobs)(delayed(_parallel_preprocess)(self, im, mask) for im, mask in zip(imgs, masks))

        train_len = dataset.train_size
        test_len = dataset.test_size
        dataset.probe.regions_train = results[:train_len]
        dataset.probe.regions_test = results[train_len:train_len + test_len]
        dataset.gallery.regions_train = results[train_len + test_len:-test_len]
        dataset.gallery.regions_test = results[-test_len:]

    def process_image(self, im, *args):
        mask = args[0]
        im_hsv = im.to_color_space(CS_HSV, normed=True)

        lineTL = np.uint16(fminbound(SilhouetteRegionsPartition.dissym_div, self.search_range_H[0], self.search_range_H[1],
                                     (im_hsv, mask, self.deltaI, self.alpha), 1e-3))
        lineHT = np.uint16(fminbound(SilhouetteRegionsPartition.dissym_div_Head, 5,
                                     lineTL, (im_hsv, mask, self.deltaI), 1e-3))

        # TODO consider subdivision
        # if self.sub_division > 1:
        #     pass
        # else:
        regions = np.asarray([(lineHT, lineTL, 0, self.J), (lineTL, self.I, 0, self.J)])
        # regions:           [       region body         ,      region legs           ]

        return regions

    @staticmethod
    def _init_sym(delta, i, img, mask):
        i = int(round(i))
        delta = int(delta)
        imgUP = img[0:i, :]
        imgDOWN = img[i - 1:, :]
        MSK_U = mask[0:i, :]
        MSK_D = mask[i - 1:, :]
        return i, delta, MSK_D, MSK_U, imgDOWN, imgUP

    @staticmethod
    def dissym_div(i, img, mask, delta, alpha):
        i, delta, MSK_D, MSK_U, imgDOWN, imgUP = SilhouetteRegionsPartition._init_sym(delta, i, img, mask)

        dimLoc = delta + 1
        indexes = list(range(dimLoc))

        imgUPloc = imgUP[indexes, :, :]
        imgDWloc = imgDOWN[indexes, :, :][:, ::-1]

        d = alpha * (1 - math.sqrt(np.sum((imgUPloc - imgDWloc) ** 2)) / dimLoc) + \
            (1 - alpha) * (abs(int(np.sum(MSK_U)) - np.sum(MSK_D)) / max([MSK_U.size, MSK_D.size]))

        return d

    @staticmethod
    def dissym_div_Head(i, img, mask, delta):
        i, delta, MSK_D, MSK_U, imgDOWN, imgUP = SilhouetteRegionsPartition._init_sym(delta, i, img, mask)

        localderU = list(range(max(i - delta, 0), i))
        localderD = list(range(0, delta + 1))

        return - abs(int(np.sum(MSK_U[localderU]))
                     - np.sum(MSK_D[localderD]))

    def dict_name(self):
        return {"Regions": str(type(self).__name__), "RegAlpha": self.alpha, "RegCount": 2 * self.sub_divisions}


class VerticalRegionsPartition(Preprocessing):
    def __init__(self, regions=None, regions_name=None):
        if regions is None:
            self.regions = [[16, 33], [33, 50], [50, 67], [67, 84], [84, 100]]
        else:
            self.regions = regions
        if regions_name is None:
            self.regions_name = "%dR" % len(self.regions)
        else:
            self.regions_name = regions_name

    def preprocess_dataset(self, dataset, n_jobs=-1):
        print("   Generating Vertical regions...")
        # [[0, 16], [16, 33], [33, 50], [50, 67], [67, 84], [84, 100]] #  Over 100% size, not actual image size
        (I, J, _) = dataset.probe.images_test[0].shape
        regions = []
        for r in self.regions:
            regions.append((int(r[0] * I / 100.), int(r[1] * I / 100.), 0, J))
        regions = np.asarray(regions)
        regions_test = [regions] * dataset.test_size
        regions_train = [regions] * dataset.train_size
        dataset.probe.regions_test = regions_test
        dataset.probe.regions_trains = regions_train
        dataset.gallery.regions_test = regions_test
        dataset.gallery.regions_trains = regions_train

    def dict_name(self):
        return {"Regions": self.regions_name, "RegCount": len(self.regions)}


class GaussianMap(Preprocessing):
    def __init__(self, alpha=0.5, kernel="GMM", sigmas=None, deviations=None):
        if sigmas is not None:
            self.sigmas = np.asarray(sigmas)
        else:
            self.sigmas = np.asarray([7.4, 8.7])
        if deviations is not None:
            self.deviations = np.asarray(deviations)
        else:
            self.deviations = np.asarray([1., 2.])
        self.J = 0
        self.deltaJ = 0
        self.search_range_V = []
        self.alpha = alpha
        self.kernel_name = kernel
        if kernel == "GMM":
            self.kernel = _gau_mix_kernel
        elif kernel == "Gaussian":
            self.kernel = _gau_kernel
        else:
            raise ValueError("Invalid kernel %s" % kernel)

    def preprocess_dataset(self, dataset, n_jobs=-1):
        print("   Calculating %s Maps..." % self.kernel_name)
        (_, self.J, _) = dataset.probe.images_test[0].shape  # Assumes all images of same shame
        self.deltaJ = self.J / 3
        self.deviations = self.sigmas / self.deviations
        self.sigmas = self.J / self.sigmas

        self.search_range_V = [self.deltaJ, (self.J - self.deltaJ) - 1]
        ims = dataset.probe.images_train + dataset.probe.images_test
        ims += dataset.gallery.images_train + dataset.gallery.images_test
        masks = dataset.probe.masks_train + dataset.probe.masks_test
        masks += dataset.gallery.masks_train + dataset.gallery.masks_test
        regions = dataset.probe.regions_train + dataset.probe.regions_test
        regions += dataset.gallery.regions_train + dataset.gallery.regions_test

        results = Parallel(n_jobs)(delayed(_parallel_preprocess)(self, i, m, r) for i, m, r in zip(ims, masks, regions))

        train_len = dataset.train_size
        test_len = dataset.test_size
        dataset.probe.maps_train = results[:train_len]
        dataset.probe.maps_test = results[train_len:train_len + test_len]
        dataset.gallery.maps_train = results[train_len + test_len:-test_len]
        dataset.gallery.maps_test = results[-test_len:]

    def process_image(self, im, *args):
        """

        :param im:
        :param args:
        :return:
        """
        mask = args[0]
        regions = args[1]

        im_hsv = im.to_color_space(CS_HSV, normed=True)

        maps = []
        for region, sigma, deviation in zip(regions, self.sigmas, self.deviations):
            lineTop = region[0]
            lineDown = region[1]
            sim_line = np.uint16(fminbound(self.sym_div, self.search_range_V[0], self.search_range_V[1],
                                           (im_hsv[lineTop:lineDown, :], mask[lineTop:lineDown, :], self.deltaJ,
                                            self.alpha), 1e-3))
            maps.append(self.kernel(sim_line, sigma, lineDown - lineTop, self.J, deviation))

        return maps

    @staticmethod
    def sym_div(i, img, mask, delta, alpha):
        i = int(round(i))
        delta = int(delta)
        imgL = img[:, 0:i]
        imgR = img[:, (i-1):]
        MSK_L = mask[:, 0:i]
        MSK_R = mask[:, (i-1):]

        dimLoc = delta + 1
        indexes = list(range(dimLoc))

        imgLloc = imgL[:, indexes, :]
        imgRloc = imgR[:, indexes, :][:, ::-1]
        d = alpha * math.sqrt(np.sum((imgLloc - imgRloc) ** 2)) / dimLoc + \
            (1 - alpha) * abs(int(np.sum(MSK_R)) - np.sum(MSK_L)) / max([MSK_L.size, MSK_R.size])
        return d

    def dict_name(self):
        return {"Map": self.kernel_name, "MapSigmas": str(self.sigmas), "MapDeviations": str(self.deviations)}


def _gau_mix_kernel(x, sigma, H, W, dev):
    x1 = float(dev)
    w1 = 0.5
    # w2 = w1
    g1 = norm.pdf(list(range(0, W)), x - x1, sigma)
    # g1 /= g1.max()
    g2 = norm.pdf(list(range(0, W)), x + x1, sigma)
    # g2 /= g2.max()
    gfinal = w1 * g1 + w1 * g2
    gfinal /= gfinal.max()
    gfinal = np.tile(gfinal, [H, 1])
    return gfinal


def _gau_kernel(x, sigma, H, W, *args):
    g = norm.pdf(list(range(0, W)), x, sigma)
    g /= g.max()
    g = np.tile(g, [H, 1])
    return g
