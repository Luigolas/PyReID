import math
import cv2
from package.utilities import InitializationError

__author__ = 'luigolas'

from sklearn.ensemble import RandomForestRegressor, RandomTreesEmbedding
import numpy as np
import package.execution as execution


class PostRankOptimization(object):
    def __init__(self):
        self.visual_expansion = RandomForestRegressor(n_estimators=200, min_samples_leaf=5, n_jobs=-1)  # As in POP
        self.probe_selected_index = None
        self.probe_selected = None  # Already feature extracted
        self.strong_negatives = []
        self.weak_negatives = []
        self.visual_expanded = []
        self.cluster_forest = RandomTreesEmbedding(n_estimators=200, min_samples_leaf=1, n_jobs=-1)  # As in POP
        self.affinity_matrix = None
        self.affinity_graph = None
        self.iteration = 0
        self.execution = None
        self.subject = 0  # The number of person Re-identified by the user

    def run(self, ex):
        self.execution = ex
        self.initial_iteration()
        while self.subject < len(self.execution.dataset.test_indexes):
            self.probe_selected_index = self.execution.dataset.test_indexes[self.subject]
            self.probe_selected = execution.probeXtest[self.subject]
            self.collage("temp.jpg")
            pass  # TODO continue

    def _generate_visual_expansion(self):
        n_estimators = self.visual_expansion.get_params()['n_estimators']
        selected_len = round(n_estimators * (2 / 3))
        selected = np.random.RandomState()
        selected = selected.permutation(n_estimators)
        selected = selected[:selected_len]
        expansion = np.empty_like(self.probe_selected)
        for i in selected:
            expansion += self.visual_expansion[i].predict(self.probe_selected)
        expansion /= float(selected_len)
        return expansion

    def calc_affinity_matrix(self, X):
        # TODO Add visual expanded elements
        leaf_indexes = self.cluster_forest.apply(X)
        n_estimators = self.cluster_forest.get_params()['n_estimators']
        affinity = np.empty((X.shape[0], X.shape[0]), np.uint16)
        # np.append(affinity, [[7, 8, 9]], axis=0)  # To add more rows later (visual expanded)
        np.fill_diagonal(affinity, n_estimators)  # Max value in diagonal
        for i in np.ndindex(affinity.shape):
            if i[0] >= i[1]:  # Already calculated (symmetric matrix)
                continue
            affinity[i] = np.sum(leaf_indexes[i[0]] == leaf_indexes[i[1]])
            affinity[i[::-1]] = affinity[i]  # Symmetric value

        return affinity

    def initial_iteration(self):
        self.visual_expansion.fit(execution.probeXtrain, execution.galleryYtrain)
        self.cluster_forest.fit(execution.galleryYtest)
        self.affinity_matrix = self.calc_affinity_matrix(execution.galleryYtest)
        # TODO Affinity graph ??

    def iterate(self):
        to_expand_len = len(self.strong_negatives) - len(self.weak_negatives)
        if to_expand_len < 0:
            raise InitializationError("There cannot be more weak negatives than strong negatives")

        for i in range(to_expand_len):
            self.visual_expanded.append(self._generate_visual_expansion())

        # TODO See what to do with visual expanded

        self.iteration += 1

    def collage(self, name, cols=5, size=10, min_gap_size=5):
        """
        Adapted from http://answers.opencv.org/question/13876/
                     read-multiple-images-from-folder-and-concat/?answer=13890#post-id-13890

        :param name: path to save collage imgf
        :param cols: num of columms for the collage
        :param size: num of images to show in collage
        :param min_gap_size: space between images
        :return:
        """
        # Add reference imgf first
        imgs = []

        img = self.execution.dataset.probe.images[self.execution.dataset.test_indexes[self.probe_selected_index]].copy()
        img[0:10, 0:10] = [0, 255, 0]
        imgs.append(img)

        elements = self.execution.ranking_matrix[self.probe_selected_index]

        # Open imgs and save in list
        size = min(len(elements), (size - 1))
        for i, elem in zip(range(size), elements):
            # print files_order_list[elem]
            img = self.execution.dataset.gallery.images[self.execution.dataset.test_indexes[elem]].copy()

            if self.execution.dataset.same_individual_by_id(self.probe_selected_index, elem, "test"):
                img[0:10, 0:10] = [0, 255, 0]

            imgs.append(img)

        # let's find out the maximum dimensions
        max_width = 0
        max_height = 0

        for img in imgs:
            max_height = max(max_height, img.shape[0])  # rows
            max_width = max(max_width, img.shape[1])  # cols

        # number of images in y direction
        rows = int(math.ceil(len(imgs) / float(cols)))

        result = np.zeros(
            (rows * max_height + (rows - 1) * min_gap_size, cols * max_width + (cols - 1) * min_gap_size, 3), np.uint8)

        current_height = current_width = i = 0

        for y in range(rows):
            for x in range(cols):
                result[current_height:current_height + imgs[i].shape[0],
                       current_width:current_width + imgs[i].shape[1]] = imgs[i]
                i += 1
                current_width += max_width + min_gap_size
            current_width = 0
            current_height += max_height + min_gap_size

        cv2.imwrite(name, result)