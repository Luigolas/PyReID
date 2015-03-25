from package.utilities import InitializationError

__author__ = 'luigolas'

from sklearn.ensemble import RandomForestRegressor, RandomTreesEmbedding
import numpy as np


class PostRankOptimization(object):
    def __init__(self, probe_list):
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
        leaf_indexes = self.cluster_forest.transform(X)
        n_estimators = self.cluster_forest.get_params()['n_estimators']
        affinity = np.empty((X.sahpe(0), X.sahpe(0)), np.uint16)
        # np.append(affinity, [[7, 8, 9]], axis=0)  # To add more rows later (visual expanded)
        np.fill_diagonal(affinity, n_estimators)  # Max value in diagonal
        for i in np.ndindex(affinity.shape):
            if i[0] >= i[1]:  # Already calculated (symmetric matrix)
                continue
            affinity[i] = np.sum(leaf_indexes[i(0)] == leaf_indexes[i(1)])
            affinity[i[::-1]] = affinity[i]  # Symmetric value

        return affinity

    def initial_iteration(self, probe_selected_index, probe_selected, X_train, Y_train, Y_test):
        self.probe_selected = probe_selected
        self.probe_selected_index = probe_selected_index
        self.visual_expansion.fit(X_train, Y_train)
        self.cluster_forest.fit(Y_test)
        self.affinity_matrix = self.calc_affinity_matrix(Y_test)
        # TODO Affinity graph ??

    def iterate(self):
        to_expand_len = len(self.strong_negatives) - len(self.weak_negatives)
        if to_expand_len < 0:
            raise InitializationError("There cannot be more weak negatives than strong negatives")

        for i in range(to_expand_len):
            self.visual_expanded.append(self._generate_visual_expansion())

        # TODO See what to do with visual expanded

        self.iteration += 1