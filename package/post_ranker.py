from copy import copy
import math
import random
import cv2
from package.utilities import InitializationError

__author__ = 'luigolas'

from sklearn.ensemble import RandomForestRegressor, RandomTreesEmbedding
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
import numpy as np
# import package.execution as execution


class PostRankOptimization(object):
    """

    :param balanced:
    :param visual_expansion_use:
    :param re_score_alpha:
    :param re_score_method_proportional:
    :param regions: Define which of the regions to be considered upper body and which legs. If None, not used.
           Must be of length 2 if defined.
           Example: regions=[[0, 1], [2, 3, 4]]
    :return:
    """
    def __init__(self, balanced=False, visual_expansion_use=True, re_score_alpha=0.15,
                 re_score_method_proportional=True, regions=None, rfr_estimators=20, rfr_leafs=5,
                 rte_estimators=20, rte_leafs=1):
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
        self.visual_expansion = RandomForestRegressor(n_estimators=rfr_estimators, min_samples_leaf=rfr_leafs,
                                                      n_jobs=-1)  # As in POP

        # regions = [[0], [1]]
        if regions is None:
            self.regions = [[0]]
            self.regions_parts = 1
        elif len(regions) == 2:
            self.regions = regions
            self.regions_parts = sum([len(e) for e in regions])
        else:
            raise ValueError("Regions size must be 2 (body region and legs region)")
        self.size_for_each_region_in_fe = 0  # Initialized at initial iteration

        # As in POP
        self.cluster_forest = [RandomTreesEmbedding(n_estimators=rte_estimators, min_samples_leaf=rte_leafs, n_jobs=-1)
                               for _ in range(len(self.regions))]

        self.affinity_matrix = []
        # self.leaf_indexes = None  # TODO ??
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
        self.feature_matcher = None  # Initialized when set_ex

    def set_ex(self, ex, rm):
        self.execution = ex
        self.ranking_matrix = rm
        self.feature_matcher = copy(self.execution.feature_matcher)
        self.feature_matcher._weights = None
        self.initial_iteration()

    def new_samples(self, weak_negatives_index, strong_negatives_index, absolute_index=False):
        self.new_weak_negatives = [[e, idx] for [e, idx] in weak_negatives_index if
                                   [e, idx] not in self.weak_negatives]
        self.new_strong_negatives = [[e, idx] for [e, idx] in strong_negatives_index if
                                     [e, idx] not in self.strong_negatives]
        if not absolute_index:
            self.new_weak_negatives = [[self.rank_list[e], idx] for [e, idx] in self.new_weak_negatives]
            self.new_strong_negatives = [[self.rank_list[e], idx] for [e, idx] in self.new_strong_negatives]

    def _generate_visual_expansion(self):
        n_estimators = self.visual_expansion.get_params()['n_estimators']
        selected_len = round(float(n_estimators) * (2 / 3.))
        selected = np.random.RandomState()
        selected = selected.permutation(n_estimators)
        selected = selected[:selected_len]
        expansion = np.zeros_like(self.probe_selected)
        for s in selected:
            expansion += self.visual_expansion[s].predict(self.probe_selected).flatten()
        expansion /= float(selected_len)
        return expansion

    def calc_affinity_matrix(self, cl_forest, X):
        # TODO Add visual expanded elements?
        leaf_indexes = cl_forest.apply(X)
        n_estimators = cl_forest.get_params()['n_estimators']
        affinity = np.empty((X.shape[0], X.shape[0]), np.uint16)
        # np.append(affinity, [[7, 8, 9]], axis=0)  # To add more rows later (visual expanded)
        np.fill_diagonal(affinity, n_estimators)  # Max value in diagonal
        for i in np.ndindex(affinity.shape):
            if i[0] >= i[1]:  # Already calculated (symmetric matrix)
                continue
            affinity[i] = np.sum(leaf_indexes[i[0]] == leaf_indexes[i[1]])
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
        self.size_for_each_region_in_fe = self.execution.dataset.gallery.fe_test.shape[1] / self.regions_parts
        if self.use_visual_expansion:
            self.visual_expansion.fit(self.execution.dataset.probe.fe_train, self.execution.dataset.gallery.fe_train)
        for idx, cl_forest in enumerate(self.cluster_forest):
            size = self.size_for_each_region_in_fe
            fe_test_idx = self.execution.dataset.gallery.fe_test[:, size * idx:size * (idx + 1)]
            cl_forest.fit(fe_test_idx)
            self.affinity_matrix.append(self.calc_affinity_matrix(cl_forest, fe_test_idx))
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
                # Randomly select if body or legs
                self.new_visual_expanded.append([self._generate_visual_expansion(), random.choice([0, 1])])

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
    """
    Based on similarity and affinity to reorder values.
    Similarity: Value calculated using Feature Matching
    Affinity: Value calculated using clustering methods
    """
    def re_score(self, sign, elem, comp_with_probe, affinity, similarity):
        """
        Calculates new comparison value for elem and updates self.comp_list[elem]
        comp_with_probe, affinity and similarity must be normalized (0..1)
        Similarity: The lowest value, the more similar (lowest distance)
        :param sign:
        :param elem:
        :param comp_with_probe: The lowest value, the more similar (lowest distance)
        :param affinity: The higher value, the more affinity
        :param elem2_fe:
        :return:
        """
        # similarity = self.execution.feature_matcher.match(elem2_fe, self.execution.dataset.gallery.fe_test[elem])
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
        for [sn, idx] in self.new_strong_negatives:
            region_size = self.size_for_each_region_in_fe * len(self.regions[idx])
            initial_pos = self.regions[idx][0] * self.size_for_each_region_in_fe
            sn_fe = self.execution.dataset.gallery.fe_test[sn][initial_pos:initial_pos + region_size]
            for elem, (elem_comp_w_probe, affinity) in enumerate(zip(self.comp_list, self.affinity_matrix[idx][sn])):
                n_estimators = self.cluster_forest[idx].get_params()['n_estimators']
                affinity = float(affinity) / n_estimators
                elem_fe = self.execution.dataset.gallery.fe_test[elem][initial_pos:initial_pos + region_size]
                similarity = self.feature_matcher.match(elem_fe, sn_fe)
                self.re_score(+1, elem, elem_comp_w_probe, affinity, similarity)

        for [wn, idx] in self.new_weak_negatives:
            region_size = self.size_for_each_region_in_fe * len(self.regions[idx])
            initial_pos = self.regions[idx][0] * self.size_for_each_region_in_fe
            wn_fe = self.execution.dataset.gallery.fe_test[wn][initial_pos:initial_pos + region_size]
            for elem, (elem_comp_w_probe, affinity) in enumerate(zip(self.comp_list, self.affinity_matrix[idx][wn])):
                n_estimators = self.cluster_forest[idx].get_params()['n_estimators']
                affinity = float(affinity) / n_estimators
                elem_fe = self.execution.dataset.gallery.fe_test[elem][initial_pos:initial_pos + region_size]
                similarity = self.feature_matcher.match(elem_fe, wn_fe)
                self.re_score(-1, elem, elem_comp_w_probe, affinity, similarity)

        for [ve, idx] in self.new_visual_expanded:
            region_size = self.size_for_each_region_in_fe * len(self.regions[idx])
            initial_pos = self.regions[idx][0] * self.size_for_each_region_in_fe
            ve_fe = ve[initial_pos:initial_pos + region_size]
            ve_cluster_value = self.cluster_forest[idx].apply(ve_fe)
            for elem, elem_comp_w_probe in enumerate(self.comp_list):
                elem_fe = self.execution.dataset.gallery.fe_test[elem][initial_pos:initial_pos + region_size]
                elem_cluster_value = self.cluster_forest[idx].apply(elem_fe)
                n_estimators = self.cluster_forest[idx].get_params()['n_estimators']
                affinity = np.sum(ve_cluster_value == elem_cluster_value)
                affinity = float(affinity) / n_estimators
                similarity = self.feature_matcher.match(elem_fe, ve_fe, )
                self.re_score(-1, elem, elem_comp_w_probe, affinity, similarity)

        self.rank_list = np.argsort(self.comp_list).astype(np.uint16)



class LabSP(PostRankOptimization):
    """
    Label Spreading method for reordering
    """
    def re_score(self, elem, proba):
        positive_proba = proba[0]
        negative_proba = proba[1]
        if positive_proba > negative_proba:
            increment = self.re_score_alpha * positive_proba
        else:
            increment = - self.re_score_alpha * negative_proba

        if self.re_score_method_proportional:
            self.comp_list[elem] += increment * self.comp_list[elem]
        else:
            # self.comp_list[elem] = ((1 - self.re_score_alpha) * comp_with_probe) + \
            #                        (sign * affinity * self.re_score_alpha)
            self.comp_list[elem] += increment

    def reorder(self):
        """
        Updates self.comp_list and self.rank_list, based on self.new_strong_negatives, self.new_weak_negatives and
          self.new_visual_expanded
        :return:
        """
        regressor = LabelSpreading(kernel='knn', alpha=0.1)  # Looks it doesn't work for rbf kernel
        # regressor = LabelPropagation(kernel='knn', alpha=0.1)
        X = self.execution.dataset.gallery.fe_test
        y = np.full((X.shape[0]), -1, np.int8)  # Default value -1

        # Positive = 1
        y[self.new_weak_negatives] = 1
        y[self.weak_negatives] = 1

        # Visual expanded = 1
        vesp = self.visual_expanded + self.new_visual_expanded
        if vesp:
            X = np.concatenate((X, vesp))
            y = np.concatenate((y, np.ones((len(vesp)))))

        # Negatives = 2
        y[self.new_strong_negatives] = 2
        y[self.strong_negatives] = 2

        regressor.fit(X, y)
        for elem, _ in enumerate(self.comp_list):
            self.re_score(elem, regressor.predict_proba(self.execution.dataset.gallery.fe_test[elem])[0])

        self.rank_list = np.argsort(self.comp_list).astype(np.uint16)