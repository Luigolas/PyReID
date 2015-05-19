import math
import cv2
from package.utilities import InitializationError

__author__ = 'luigolas'

from sklearn.ensemble import RandomForestRegressor, RandomTreesEmbedding
import numpy as np
# import package.execution as execution


class PostRankOptimization(object):
    def __init__(self, balanced=False, visual_expansion_use=True, re_score_alpha=0.15,
                 re_score_method_proportional=True):
        self.subject = -1  # The order of the person to be Re-identified by the user (Initially -1)
        self.probe_name = ""
        self.probe_selected = None  # Already feature extracted
        self.target_position = 0
        self.iteration = 0
        self.strong_negatives = []
        self.weak_negatives = []
        self.visual_expanded = []
        self.new_strong_negatives = []
        self.new_weak_negatives = []
        self.new_visual_expanded = []
        self.visual_expansion = RandomForestRegressor(n_estimators=20, min_samples_leaf=5, n_jobs=-1)  # As in POP
        self.cluster_forest = RandomTreesEmbedding(n_estimators=20, min_samples_leaf=1, n_jobs=-1)  # As in POP
        self.affinity_matrix = None
        self.leaf_indexes = None
        # self.affinity_graph = None
        self.execution = None
        self.ranking_matrix = None
        self.rank_list = None
        self.comp_list = None
        self.balanced = balanced
        if not balanced:
            self.use_visual_expansion = False
        else:
            self.use_visual_expansion = visual_expansion_use
        self.re_score_alpha = re_score_alpha
        self.re_score_method_proportional = re_score_method_proportional

    def set_ex(self, ex, rm):
        self.execution = ex
        self.ranking_matrix = rm
        self.initial_iteration()

    def new_samples(self, weak_negatives_index, strong_negatives_index):
        self.new_weak_negatives = [e for e in self.rank_list[weak_negatives_index] if e not in self.weak_negatives]
        self.new_strong_negatives = [e for e in self.rank_list[strong_negatives_index] if
                                     e not in self.strong_negatives]

    def _generate_visual_expansion(self):
        n_estimators = self.visual_expansion.get_params()['n_estimators']
        selected_len = round(float(n_estimators) * (2 / 3.))
        selected = np.random.RandomState()
        selected = selected.permutation(n_estimators)
        selected = selected[:selected_len]
        expansion = np.zeros_like(self.probe_selected)
        for i in selected:
            expansion += self.visual_expansion[i].predict(self.probe_selected).flatten()
        expansion /= float(selected_len)
        return expansion

    def calc_affinity_matrix(self, X):
        # TODO Add visual expanded elements?
        self.leaf_indexes = self.cluster_forest.apply(X)
        n_estimators = self.cluster_forest.get_params()['n_estimators']
        affinity = np.empty((X.shape[0], X.shape[0]), np.uint16)
        # np.append(affinity, [[7, 8, 9]], axis=0)  # To add more rows later (visual expanded)
        np.fill_diagonal(affinity, n_estimators)  # Max value in diagonal
        for i in np.ndindex(affinity.shape):
            if i[0] >= i[1]:  # Already calculated (symmetric matrix)
                continue
            affinity[i] = np.sum(self.leaf_indexes[i[0]] == self.leaf_indexes[i[1]])
            affinity[i[::-1]] = affinity[i]  # Symmetric value

        return affinity

    def new_subject(self):
        if self.subject < self.execution.dataset.test_size:
            self.subject += 1
            self.probe_name = self.execution.dataset.probe.files_test[self.subject]
            self.probe_name = "/".join(self.probe_name.split("/")[-2:])
            self.probe_selected = self.execution.dataset.probe.fe_test[self.subject]
            self.rank_list = self.ranking_matrix[self.subject].copy()
            self.comp_list = self.execution.matching_matrix[self.subject].copy()
            self._calc_target_position()
            self.iteration = 0
            self.strong_negatives = []
            self.weak_negatives = []
            self.visual_expanded = []
        else:
            return  # TODO Control situation

    def initial_iteration(self):
        self.new_subject()
        if self.use_visual_expansion:
            self.visual_expansion.fit(self.execution.dataset.probe.fe_train, self.execution.dataset.gallery.fe_train)
        self.cluster_forest.fit(self.execution.dataset.gallery.fe_test)
        self.affinity_matrix = self.calc_affinity_matrix(self.execution.dataset.gallery.fe_test)
        # TODO Affinity graph ??

    def iterate(self):
        self.iteration += 1
        print("Iteration %d" % self.iteration)
        to_expand_len = len(self.new_strong_negatives) - len(self.new_weak_negatives)
        if self.balanced:
            if to_expand_len < 0:
                return "There cannot be more weak negatives than strong negatives"
            elif to_expand_len > 0 and not self.use_visual_expansion:
                return "There must be the same number of weak negatives and strong negatives"

            for i in range(to_expand_len):
                self.new_visual_expanded.append(self._generate_visual_expansion())

        self.reorder()

        self._calc_target_position()

        self.strong_negatives.extend(self.new_strong_negatives)
        self.weak_negatives.extend(self.new_weak_negatives)
        self.visual_expanded.extend(self.new_visual_expanded)
        self.new_strong_negatives = []
        self.new_weak_negatives = []
        self.new_visual_expanded = []
        return "OK"

    def collage(self, name, cols=5, size=20, min_gap_size=5):
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

        img = self.execution.dataset.probe.images_test[self.subject].copy()
        img[0:10, 0:10] = [0, 255, 0]
        cv2.putText(img, "Probe", (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
        imgs.append(img)

        elements = self.rank_list.copy()

        # Open imgs and save in list
        size = min(len(elements), (size - 1))
        for i, elem in zip(range(size), elements):
            # print files_order_list[elem]
            img = self.execution.dataset.gallery.images_test[elem].copy()
            if self.execution.dataset.same_individual_by_pos(self.subject, elem, "test"):
                img[0:10, 0:10] = [0, 255, 0]
            cv2.putText(img, str(i), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 2)

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
        cv2.imshow("tal", result)
        cv2.waitKey(1)

    def reorder(self):
        raise NotImplementedError("Please Implement reorder method")

    def _calc_target_position(self):
        for column, elemg in enumerate(self.rank_list):
            if self.execution.dataset.same_individual_by_pos(self.subject, elemg, selected_set="test"):
                target_position = column  # TODO: If not multiview we could exit loop here
                self.target_position = target_position
                break


class SAA(PostRankOptimization):
    def re_score(self, sign, elem, comp_with_probe, affinity, elem2_fe):
        """
        comp_with_probe, affinity and similarity must be normalized (0..1)
        Similarity: The lowest value, the more similar (lowest distance)
        :param sign:
        :param elem:
        :param comp_with_probe: The lowest value, the more similar (lowest distance)
        :param affinity: The higher value, the more affinity
        :param elem2_fe:
        :return:
        """
        similarity = self.execution.feature_matcher.match(elem2_fe, self.execution.dataset.gallery.fe_test[elem])
        increment = sign * self.re_score_alpha
        if self.re_score_method_proportional:
            self.comp_list[elem] = comp_with_probe + (increment * comp_with_probe *
                                                      (0.5 * affinity + 0.5 * (1 - similarity)))
        else:
            # self.comp_list[elem] = ((1 - self.re_score_alpha) * comp_with_probe) + \
            #                        (sign * affinity * self.re_score_alpha)
            self.comp_list[elem] = ((1 - self.re_score_alpha) * comp_with_probe) + \
                                   (increment * (0.5 * affinity + 0.5 * (1 - similarity)))

    def reorder(self):
        for sn in self.new_strong_negatives:
            for elem, (comp_with_probe, affinity) in enumerate(zip(self.comp_list, self.affinity_matrix[sn])):
                n_estimators = self.cluster_forest.get_params()['n_estimators']
                affinity = float(affinity) / n_estimators
                self.re_score(+1, elem, comp_with_probe, affinity, self.execution.dataset.gallery.fe_test[sn])

        for wn in self.new_weak_negatives:
            for elem, (comp_with_probe, affinity) in enumerate(zip(self.comp_list, self.affinity_matrix[wn])):
                n_estimators = self.cluster_forest.get_params()['n_estimators']
                affinity = float(affinity) / n_estimators
                self.re_score(-1, elem, comp_with_probe, affinity, self.execution.dataset.gallery.fe_test[wn])

        for ve in self.new_visual_expanded:
            ve_cluster_value = self.cluster_forest.apply(ve)
            for elem, comp_with_probe in enumerate(self.comp_list):
                elem_cluster_value = self.leaf_indexes[elem]
                n_estimators = self.cluster_forest.get_params()['n_estimators']
                affinity = np.sum(ve_cluster_value == elem_cluster_value)
                affinity = float(affinity) / n_estimators
                self.re_score(-1, elem, comp_with_probe, affinity, ve)

        self.rank_list = np.argsort(self.comp_list).astype(np.uint16)


