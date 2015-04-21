__author__ = 'luigolas'

import os
from package.image import Image
from package.utilities import ImagesNotFoundError, NotADirectoryError


class ImageSet(object):
    def __init__(self, folder_name, name_ids=2):
        self.path = ImageSet._valid_directory(folder_name)
        self.name = "_".join(self.path.split("/")[-name_ids:])
        # name = "_".join(name)
        self.files = self._read_all_files()
        self.dataset_len = len(self.files)
        if self.dataset_len == 0:
            raise ImagesNotFoundError("At folder " + self.path)
        # self.images = self.load_images()
        self.images_train = []
        self.images_test = []
        self.masks_train = []
        self.masks_test = []
        self.files_train = []
        self.files_test = []
        self.regions_train = []
        self.regions_test = []
        self.maps_train = []
        self.maps_test = []
        self.fe_train = []
        self.fe_test = []

    def _read_all_files(self):
        files = []
        for path, subdirs, files_order_list in os.walk(self.path):
            for filename in files_order_list:
                if ImageSet._valid_format(filename):
                    f = os.path.join(path, filename)
                    files.append(f)
        return files

    def load_images(self):
        self.images_train = []
        self.images_test = []

        if not self.files_test:  # If not initialized
            self.files_test = self.files
            self.files_train = []

        for imname in self.files_train:
            self.images_train.append(Image.from_filename(imname))

        for imname in self.files_test:
            self.images_test.append(Image.from_filename(imname))


    @staticmethod
    def _valid_format(name):
        return ((".jpg" in name) or (".png" in name) or (
            ".bmp" in name)) and "MASK" not in name and "FILTERED" not in name

    @staticmethod
    def _valid_directory(folder_name):
        if not os.path.isdir(folder_name):
            raise NotADirectoryError("Not a valid directory path: " + folder_name)
        if folder_name[-1] == '/':
            folder_name = folder_name[:-1]
        return folder_name

    def unload(self):
        self.images_train = None
        self.images_train = None
        self.masks_train = None
        self.masks_test = None
        self.files = None
        self.files_train = None
        self.files_test = None

