__author__ = 'luigolas'


def set_preprocessing(self, preproc, w_masks=True):
    self._preprocessing = preproc
    self._preprocessing_w_masks = w_masks

def set_preprocessing_w_masks(self, w_masks):
    self._preprocessing_w_masks = w_masks

def preprocess(self):
    if self.probe.images is None:
        self.probe.load_images()
    if self.gallery.images is None:
        self.gallery.load_images()

    if self._preprocessing is None:
        return

    f = None
    if self._preprocessing == "CBTF":
        f = btf(self.probe.images, self.gallery.images, self.probe.masks, self.gallery.masks)
    elif self._preprocessing == "gMBTF":
        elements_left = list(range(self.probe.len))
        btfs = [np.array([0] * 256), np.array([0] * 256), np.array([0] * 256)]
        count_btfs = 0
        while len(elements_left) > 0:
            individual = [elements_left.pop(0)]
            aux_list = []
            for elem in elements_left:
                if self.same_individual(self.probe.files[individual[0]], self.probe.files[elem]):
                    individual.append(elem)
                else:
                    aux_list.append(elem)
            elements_left = aux_list
            to_compare = [self.gallery.files.index(x) for x in self.gallery.files
                          if self.same_individual(self.gallery.files[individual[0]], x)]
            # Load images
            individual_images = [self.probe.images[x] for x in individual]
            to_compare_images = [self.gallery.images[x] for x in to_compare]
            masks1 = None
            masks2 = None
            if self.probe.masks is not None:
                masks1 = [self.probe.masks[x] for x in individual]
            if self.probe.masks is not None:
                masks2 = [self.gallery.masks[x] for x in to_compare]
            result = btf(individual_images, to_compare_images, masks1, masks2)
            count_btfs += 1
            for channel, elem in enumerate(result):
                btfs[channel] += elem
        f = [np.asarray(np.rint(x / count_btfs), np.int) for x in btfs]

    elif self._preprocessing == "ngMBTF":
        btfs = [np.array([0] * 256), np.array([0] * 256), np.array([0] * 256)]
        count_btfs = 0
        for index, im in enumerate(self.probe.images):
            # Keep index of images to compare to
            to_compare = [self.gallery.files.index(x) for x in self.gallery.files
                          if self.same_individual(im.imgname, x)]
            for im2compare in to_compare:
                mask1 = None
                mask2 = None
                im2 = self.gallery.images[im2compare]
                if self.probe.masks is not None:
                    mask1 = self.probe.masks[index]
                if self.gallery.masks is not None:
                    mask2 = self.gallery.masks[im2compare]
                # btfs.append(btf(im, im2, mask1, mask2))
                result = btf(im, im2, mask1, mask2)
                count_btfs += 1
                for channel, elem in enumerate(result):
                    btfs[channel] += elem
        f = [np.asarray(np.rint(x / count_btfs), np.int) for x in btfs]

    else:
        raise AttributeError("Not a valid preprocessing key")

    if f is None:
        raise NotImplementedError
    newimages = []
    for im in self.probe.images:
        newimages.append(convert_image(f, im, self._preprocessing))
    self.probe.images = newimages

def btf(im1, im2, mask1=None, mask2=None):
    def find_nearest(array, value):
        return (np.abs(array - value)).argmin()

    cumh1 = cummhist(im1, masks=mask1)
    cumh2 = cummhist(im2, masks=mask2)
    # For each value in cumh1, look for the closest one (floor, ceil, round?) in cum2, and save index of cum2.
    func = [np.empty_like(h, np.uint8) for h in cumh1]
    for f_i, hist_i, hist2_i in zip(func, cumh1, cumh2):  # For each channel
        for index, value in enumerate(hist_i):
            f_i[index] = find_nearest(hist2_i, value)
    return func

def cummhist(ims, colorspace=image.CS_BGR, masks=None):
    ev = evaluator.Histogram(colorspace, None, None, "2D")

    if type(ims) is image.Image:
        ims = [ims]
    if type(masks) is not list:
        masks = [masks] * len(ims)
    h = []
    for im, mask in zip(ims, masks):
        result = ev.evaluate(im, mask, normalization=None)[0]
        h = [a + b for a, b in itertools.zip_longest(h, result, fillvalue=0)]  # Accumulate with previous histograms

    # Normalize each histogram
    return [evaluator.normalizeHist(h_channel.cumsum(), normalization=cv2.NORM_INF) for h_channel in h]

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