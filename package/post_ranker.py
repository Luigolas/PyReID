from copy import copy
import math
import random
import cv2
from package.utilities import InitializationError
from sklearn.ensemble import RandomForestRegressor, RandomTreesEmbedding
from sklearn.semi_supervised import LabelSpreading, LabelPropagation
import numpy as np
from profilehooks import profile

__author__ = 'luigolas'


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
                 re_score_proportional=True, regions=None, ve_estimators=20, ve_leafs=5):  # OK
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
        self.visual_expansion = RandomForestRegressor(n_estimators=ve_estimators, min_samples_leaf=ve_leafs,
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
        self.re_score_proportional = re_score_proportional

    def set_ex(self, ex, rm):  # OK
        self.execution = ex
        self.ranking_matrix = rm
        self.initial_iteration()

    def new_samples(self, weak_negatives_index, strong_negatives_index, absolute_index=False):  # OK
        self.new_weak_negatives = [[e, idx] for [e, idx] in weak_negatives_index if
                                   [e, idx] not in self.weak_negatives]
        self.new_strong_negatives = [[e, idx] for [e, idx] in strong_negatives_index if
                                     [e, idx] not in self.strong_negatives]
        if not absolute_index:
            self.new_weak_negatives = [[self.rank_list[e], idx] for [e, idx] in self.new_weak_negatives]
            self.new_strong_negatives = [[self.rank_list[e], idx] for [e, idx] in self.new_strong_negatives]

    def _generate_visual_expansion(self):  # OK
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

    def new_subject(self):  # OK
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

    def initial_iteration(self):  # OK
        self.new_subject()
        self.size_for_each_region_in_fe = self.execution.dataset.gallery.fe_test.shape[1] / self.regions_parts
        if self.use_visual_expansion:
            self.visual_expansion.fit(self.execution.dataset.probe.fe_train, self.execution.dataset.gallery.fe_train)

    def iterate(self):  # OK
        self.iteration += 1
        # print("Iteration %d" % self.iteration)
        to_expand_len = len(self.new_strong_negatives) - len(self.new_weak_negatives)
        if self.balanced:
            if to_expand_len < 0:
                return "There cannot be more weak negatives than strong negatives"
            elif to_expand_len > 0 and not self.use_visual_expansion:
                return "There must be the same number of weak negatives and strong negatives"

            for i in range(to_expand_len):
                # Randomly select if body or legs
                if len(self.regions) == 1:
                    reg = 0
                else:  # Assumes only two body parts
                    reg = random.choice([0, 1])
                self.new_visual_expanded.append([self._generate_visual_expansion(), reg])

        self.reorder()

        self._calc_target_position()

        self.strong_negatives.extend(self.new_strong_negatives)
        self.weak_negatives.extend(self.new_weak_negatives)
        self.visual_expanded.extend(self.new_visual_expanded)
        self.new_strong_negatives = []
        self.new_weak_negatives = []
        self.new_visual_expanded = []
        return "OK"

    def collage(self, name, cols=5, size=20, min_gap_size=5):  # OK
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

    def reorder(self):  # OK
        raise NotImplementedError("Please Implement reorder method")

    def _calc_target_position(self):  # OK
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
        :param similarity:
        :return:
        """
        # similarity = self.execution.feature_matcher.match(elem2_fe, self.execution.dataset.gallery.fe_test[elem])
        increment = sign * self.re_score_alpha
        sim_beta = 1 - self.affinity_beta
        if self.re_score_proportional:
            self.comp_list[elem] = comp_with_probe + (increment * comp_with_probe *
                                                      (self.affinity_beta * affinity + sim_beta * (1 - similarity)))
        else:
            # self.comp_list[elem] = ((1 - self.re_score_alpha) * comp_with_probe) + \
            #                        (sign * affinity * self.re_score_alpha)
            self.comp_list[elem] = ((1 - self.re_score_alpha) * comp_with_probe) + \
                                   (increment * (self.affinity_beta * affinity + sim_beta * (1 - similarity)))

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
            X = self.execution.dataset.gallery.fe_test[:, initial_pos:initial_pos + region_size]
            leaf_indexes = self.cluster_forest[idx].apply(X)
            for elem, elem_comp_w_probe in enumerate(self.comp_list):
                elem_fe = self.execution.dataset.gallery.fe_test[elem][initial_pos:initial_pos + region_size]
                elem_cluster_value = leaf_indexes[elem]
                n_estimators = self.cluster_forest[idx].get_params()['n_estimators']
                affinity = np.sum(ve_cluster_value == elem_cluster_value)
                affinity = float(affinity) / n_estimators
                similarity = self.feature_matcher.match(elem_fe, ve_fe)
                self.re_score(-1, elem, elem_comp_w_probe, affinity, similarity)

        self.rank_list = np.argsort(self.comp_list).astype(np.uint16)


class LabSP(PostRankOptimization):
    """
    Label Spreading method for reordering
    """

    def __init__(self, method="spreading", kernel="knn", alpha=0.2, gamma=20, n_neighbors=7, **kwargs):
        super(LabSP, self).__init__(**kwargs)
        if method.lower() == "propagation":
            self.regressors = [LabelPropagation(kernel=kernel, alpha=alpha, gamma=gamma, n_neighbors=n_neighbors)
                               for _ in range(len(self.regions))]
        elif method.lower() == "spreading":
            self.regressors = [LabelSpreading(kernel=kernel, alpha=alpha, gamma=gamma, n_neighbors=n_neighbors)
                               for _ in range(len(self.regions))]
        else:
            raise InitializationError("Method %s not valid" % method)

    def re_score(self, elem, proba):
        # try:
        #     proba[1]
        # except IndexError:
        #     print("elem is %d" % elem)
        positive_proba = proba[0]
        negative_proba = proba[1]
        if positive_proba > negative_proba:
            increment = self.re_score_alpha * positive_proba
        else:
            increment = - self.re_score_alpha * negative_proba

        if self.re_score_proportional:
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
        X = self.execution.dataset.gallery.fe_test
        y = np.full((X.shape[0], len(self.regressors)), -1, np.int8)  # Default value -1

        # Positive = 1
        if self.new_weak_negatives:
            new_weak = np.array(self.new_weak_negatives)
            y[new_weak[:, 0], new_weak[:, 1]] = 1
        if self.weak_negatives:
            weak = np.array(self.weak_negatives)
            y[weak[:, 0], weak[:, 1]] = 1

        # Visual expanded = 1
        vesp = self.visual_expanded + self.new_visual_expanded
        if vesp:
            X = np.concatenate((X, [e[0] for e in vesp]))
            vals = np.full((len(vesp), len(self.regions)), -1, np.int8)
            vals[range(vals.shape[0]), [e[1] for e in vesp]] = 1
            y = np.concatenate((y, vals))

        # Negatives = 2
        if self.new_strong_negatives:
            new_strong = np.array(self.new_strong_negatives)
            y[new_strong[:, 0], new_strong[:, 1]] = 2
        if self.strong_negatives:
            strong = np.array(self.strong_negatives)
            y[strong[:, 0], strong[:, 1]] = 2

        for idx, regressor in enumerate(self.regressors):
            region_size = self.size_for_each_region_in_fe * len(self.regions[idx])
            initial_pos = self.regions[idx][0] * self.size_for_each_region_in_fe
            Xr = X[:, initial_pos:region_size]
            yr = y[:, idx]
            if yr.max() == -1:
                continue
            else:
                self.regressors[idx].fit(Xr, yr)

        X = self.execution.dataset.gallery.fe_test

        for idx, regressor in enumerate(self.regressors):
            if not hasattr(regressor, "n_iter_"):  # check if initialized
                continue
            region_size = self.size_for_each_region_in_fe * len(self.regions[idx])
            initial_pos = self.regions[idx][0] * self.size_for_each_region_in_fe
            Xr = X[:, initial_pos:region_size]
            for elem in range(len(self.comp_list)):
                self.re_score(elem, regressor.predict_proba(Xr[elem])[0])

        self.rank_list = np.argsort(self.comp_list).astype(np.uint16)


class SAL(PostRankOptimization):
    """
    Based on similarity, affinity and Label Propagation to reorder values.
    Similarity: Value calculated using Feature Matching
    Affinity: Value calculated using clustering methods
    Label: Value calculated using LabelPropagation/Spreading
    """

    def __init__(self, balanced=False, visual_expansion_use=True, re_score_alpha=0.15,
                 re_score_proportional=True, regions=None, ve_estimators=20, ve_leafs=5, clf_estimators=20,
                 clf_leafs=1, method="spreading", kernel="knn", lab_alpha=0.2, lab_gamma=20, lab_n_neighbors=7,
                 weights=None):
        super(SAL, self).__init__(balanced=balanced, visual_expansion_use=visual_expansion_use,
                                  re_score_alpha=re_score_alpha,
                                  re_score_proportional=re_score_proportional,
                                  regions=regions, ve_estimators=ve_estimators, ve_leafs=ve_leafs)
        if not weights:
            # self.weights = [0.45, 0.45, 0.1]
            self.weights = [0.5, 0.5, 0.]
        else:
            self.weights = weights

        self.cluster_forest = [RandomTreesEmbedding(n_estimators=clf_estimators, min_samples_leaf=clf_leafs, n_jobs=-1)
                               for _ in range(len(self.regions))]
        self.affinity_matrix = []
        self.feature_matcher = None  # Initialized when set_ex
        if method.lower() == "propagation":
            self.regressors = [
                LabelPropagation(kernel=kernel, alpha=lab_alpha, gamma=lab_gamma, n_neighbors=lab_n_neighbors)
                for _ in range(len(self.regions))]
        elif method.lower() == "spreading":
            self.regressors = [
                LabelSpreading(kernel=kernel, alpha=lab_alpha, gamma=lab_gamma, n_neighbors=lab_n_neighbors)
                for _ in range(len(self.regions))]
        else:
            raise InitializationError("Method %s not valid" % method)

    def set_ex(self, ex, rm):
        super(SAL, self).set_ex(ex, rm)
        self.feature_matcher = copy(self.execution.feature_matcher)
        self.feature_matcher._weights = None

    def initial_iteration(self):
        super(SAL, self).initial_iteration()
        for idx, cl_forest in enumerate(self.cluster_forest):
            region_size = self.size_for_each_region_in_fe * len(self.regions[idx])
            initial_pos = self.regions[idx][0] * self.size_for_each_region_in_fe
            fe_test_idx = self.execution.dataset.gallery.fe_test[:, initial_pos:region_size]
            cl_forest.fit(fe_test_idx)
            self.affinity_matrix.append(self.calc_affinity_matrix(cl_forest, fe_test_idx))

    def init_regressors(self, region_size_list, initial_pos_list):
        X = self.execution.dataset.gallery.fe_test
        y = np.full((X.shape[0], len(self.regressors)), -1, np.int8)  # Default value -1

        # Positive = 1
        if self.new_weak_negatives:
            new_weak = np.array(self.new_weak_negatives)
            y[new_weak[:, 0], new_weak[:, 1]] = 1
        if self.weak_negatives:
            weak = np.array(self.weak_negatives)
            y[weak[:, 0], weak[:, 1]] = 1

        # Visual expanded = 1
        vesp = self.visual_expanded + self.new_visual_expanded
        if vesp:
            X = np.concatenate((X, [e[0] for e in vesp]))
            vals = np.full((len(vesp), len(self.regions)), -1, np.int8)
            vals[range(vals.shape[0]), [e[1] for e in vesp]] = 1
            y = np.concatenate((y, vals))

        # Negatives = 2
        if self.new_strong_negatives:
            new_strong = np.array(self.new_strong_negatives)
            y[new_strong[:, 0], new_strong[:, 1]] = 2
        if self.strong_negatives:
            strong = np.array(self.strong_negatives)
            y[strong[:, 0], strong[:, 1]] = 2

        for idx, regressor in enumerate(self.regressors):
            region_size = region_size_list[idx]
            initial_pos = initial_pos_list[idx]
            Xr = X[:, initial_pos:initial_pos + region_size]
            yr = y[:, idx]
            if yr.max() == -1:
                continue
            else:
                self.regressors[idx].fit(Xr, yr)

    @staticmethod
    def calc_affinity_matrix(cl_forest, X):
        # TODO Add visual expanded elements?
        leaf_indexes = cl_forest.apply(X)
        n_estimators = cl_forest.get_params()['n_estimators']
        affinity = np.empty((X.shape[0], X.shape[0]), np.uint16)
        # affinity = np.zeros((X.shape[0], X.shape[0]), np.uint16)
        # np.append(affinity, [[7, 8, 9]], axis=0)  # To add more rows later (visual expanded)
        np.fill_diagonal(affinity, n_estimators)  # Max value in diagonal
        for i1, i2 in zip(*np.triu_indices(affinity.shape[0], 1, affinity.shape[0])):
            # for i in np.ndindex(affinity.shape):
            #     if i[0] >= i[1]:  # Already calculated (symmetric matrix)
            #         continue
            affinity[i1, i2] = np.sum(leaf_indexes[i1] == leaf_indexes[i2])
            affinity[i2, i1] = affinity[i1, i2]  # Symmetric value

        return affinity / float(n_estimators)

    def re_score(self, similarity, affinity, lab_score):

        if self.re_score_proportional:
            # self.comp_list[elem] += elem_simil_w_probe * self.re_score_alpha * (self.weights[0] * similarity[elem] +
            #                                                                     self.weights[1] * affinity[elem] +
            #                                                                     self.weights[2] * lab_score[elem])
            self.comp_list += self.comp_list * self.re_score_alpha * (self.weights[0] * similarity +
                                                                      self.weights[1] * affinity +
                                                                      self.weights[2] * lab_score)
        else:
            self.comp_list += self.re_score_alpha * (self.weights[0] * similarity +
                                                     self.weights[1] * affinity +
                                                     self.weights[2] * lab_score)

    # @profile()
    def reorder(self):
        region_size_list = []
        initial_pos_list = []
        elems_fe_list = []
        for region in range(len(self.regions)):
            region_size_list.append(self.size_for_each_region_in_fe * len(self.regions[region]))
            initial_pos_list.append(self.regions[region][0] * self.size_for_each_region_in_fe)
            region_size = region_size_list[region]
            initial_pos = initial_pos_list[region]
            elems_fe_list.append(self.execution.dataset.gallery.fe_test[:, initial_pos:initial_pos + region_size])

        similarity_sn = self._similarity(elems_fe_list, self.new_strong_negatives, region_size_list, initial_pos_list)
        similarity_wn = self._similarity(elems_fe_list, self.new_weak_negatives + self.new_visual_expanded,
                                         region_size_list, initial_pos_list)

        affinity_sn, affinity_wn = self._affinity(elems_fe_list, region_size_list, initial_pos_list)

        if self.weights[2] > 0:
            self.init_regressors(region_size_list, initial_pos_list)
            lab_score = self._labscore(elems_fe_list)
        else:
            lab_score = np.asarray([0] * len(self.comp_list))

        self.re_score(similarity_wn - similarity_sn, affinity_wn - affinity_sn, lab_score)
        self.rank_list = np.argsort(self.comp_list).astype(np.uint16)

    def _affinity(self, elems_fe_list, region_size_list, initial_pos_list):
        n_estimators = self.cluster_forest[0].get_params()['n_estimators']  # Assumes all forests have same size
        affinity_sn = []
        for [sn, region] in self.new_strong_negatives:
            affinity_sn.append(self.affinity_matrix[region][:, sn])
        affinity_sn = np.asarray(affinity_sn).mean(axis=0)

        affinity_wn = []
        for [wn, region] in self.new_weak_negatives:
            affinity_wn.append(self.affinity_matrix[region][:, wn])

        if self.new_visual_expanded:
            elems_cluster_value = [self.cluster_forest[region].apply(elems_fe_list[region])
                                   for region in range(len(self.regions))]
        for [ve, region] in self.new_visual_expanded:
            region_size = region_size_list[region]
            initial_pos = initial_pos_list[region]
            ve_fe = ve[initial_pos:initial_pos + region_size]
            ve_cluster_value = self.cluster_forest[region].apply(ve_fe)
            affinity_ve = [np.sum(ve_cluster_value == elem_cl_value) / float(n_estimators)
                           for elem_cl_value in elems_cluster_value[region]]
            affinity_wn.append(np.asarray(affinity_ve))
        affinity_wn = np.asarray(affinity_wn).mean(axis=0)
        return affinity_sn, affinity_wn

    def _similarity(self, elems_fe_list, samples, region_size_list, initial_pos_list):
        samples_fe_list = [[] for _ in range(len(self.regions))]  # Separate fe by regions
        for [sample, region] in samples:
            region_size = region_size_list[region]
            initial_pos = initial_pos_list[region]
            if type(sample) == np.ndarray:
                samples_fe_list[region].append(sample[initial_pos:initial_pos + region_size])
            else:
                samples_fe_list[region].append(self.execution.dataset.gallery.fe_test[sample]
                                               [initial_pos:initial_pos + region_size])

        similarity_samples = [[] for _ in range(len(self.regions))]
        for region in range(len(self.regions)):
            if samples_fe_list[region]:
                samples_fe_list[region] = np.asarray(samples_fe_list[region])
                similarity_samples[region] = self.feature_matcher.match_sets(elems_fe_list[region],
                                                                             samples_fe_list[region], n_jobs=-1,
                                                                             verbosity=0)
                similarity_samples[region] = similarity_samples[region].mean(axis=1)
            else:
                similarity_samples[region] = None
        similarity_samples = [e for e in similarity_samples if e is not None]
        return np.asarray(similarity_samples).mean(axis=0)

    def calc_affinity(self, elem):
        n_estimators = self.cluster_forest[0].get_params()['n_estimators']  # Assumes all forests have same size
        affinity_sn = []
        for [sn, region] in self.new_strong_negatives:
            affinity_sn.append(self.affinity_matrix[region][elem][sn])

        affinity_wn = []
        for [wn, region] in self.new_weak_negatives:
            affinity_wn.append(self.affinity_matrix[region][elem][wn])

        for [ve, region] in self.new_visual_expanded:
            region_size = self.size_for_each_region_in_fe * len(self.regions[region])
            initial_pos = self.regions[region][0] * self.size_for_each_region_in_fe
            ve_fe = ve[initial_pos:initial_pos + region_size]
            elem_fe = self.execution.dataset.gallery.fe_test[elem][initial_pos:initial_pos + region_size]
            ve_cluster_value = self.cluster_forest[region].apply(ve_fe)
            elem_cluster_value = self.cluster_forest[region].apply(elem_fe)
            affinity = np.sum(ve_cluster_value == elem_cluster_value)
            affinity = float(affinity) / n_estimators
            affinity_wn.append(affinity)

        if not affinity_sn:
            affinity_sn = [0]
        if not affinity_wn:
            affinity_wn = [0]

        affinity_wn = np.asarray(affinity_wn).mean()
        affinity_sn = np.asarray(affinity_sn).mean()
        return affinity_wn - affinity_sn

    def _labscore(self, elems_fe_list):
        predictions = []
        for idx, regressor in enumerate(self.regressors):
            if not hasattr(regressor, "n_iter_"):  # check if initialized
                predictions.append(None)
            Xr = elems_fe_list[idx]
            pred = regressor.predict_proba(Xr)
            pred = pred[:, 0] - pred[:, 1]
            predictions.append(pred)
        predictions = [e for e in predictions if e is not None]
        return np.asarray(predictions).mean(axis=0)
