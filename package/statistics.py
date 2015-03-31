import os

__author__ = 'luigolas'

from package.utilities import InitializationError
import numpy as np
from matplotlib import pyplot as plt
import itertools


class Statistics():
    """
    Position List: for each element in probe, find its same ids in gallery. Format: np.array([[2,14],[1,2],...])
    Mean List: Calculate means of position list by axis 0. Format: np.array([1.52, 4.89])
    Mode_list: TODO******* calculate statistical mode along axis 0. Format: (np.array([[2., 4.]]), np.array([[10, 12]]))
    Prob admissible range: For each row in Position List calculates if it is lower than a value. Then sum and calculates
                           percentage
    """
    def __init__(self, execution=None, position_list_set=False, mean_list_set=False, rangeX_set=None,
                 autoplotdir=None):
        if not rangeX_set:
            rangeX_set = []

        self._exec = execution
        self._mean_list_set = mean_list_set
        self._rangeX_set = rangeX_set
        if mean_list_set or rangeX_set:
            self._position_list_set = True
        else:
            self._position_list_set = position_list_set
        if autoplotdir == "": autoplotdir = "."
        self._autoplotdir = autoplotdir

        self.position_list = None
        self.mean_list = None
        self.rangeX = []
        self._name = None

    # Setter methods
    def set_execution(self, execution):
        self._exec = execution
        self._name = self._exec.name()

    def set_position_list(self):
        self._position_list_set = True

    def set_mean_list(self):
        self._mean_list_set = True
        self._position_list_set = True

    def set_rangeX(self, values):
        self._rangeX_set = values
        self._position_list_set = True

    def name(self):
        if self._name is not None:
            return self._name
        else:
            return self._exec.name()

    def dict_name(self):
        name = self._exec.dict_name()

        if self.mean_list is None:
            name.update({"MeanList1": np.NaN, "MeanList2": np.NaN})
        else:
            name.update({"MeanList1": self.mean_list[0]})
            if len(self.mean_list) > 1:
                name.update({"MeanList2": self.mean_list[1]})

        if self.rangeX is not None:
            for rangenum, range in zip(self._rangeX_set, self.rangeX):
                for index, range_elem in enumerate(range):
                    name.update({"Range-%d-%d" % (rangenum, index): range_elem})
        else:
            pass
            # name.update({"ProbAdmissible1": np.NaN, "ProbAdmissible2": np.NaN})

        return name

    def run(self):
        self._check_initialization()
        if self._position_list_set:
            self._calc_position_list()
        if self._mean_list_set:
            self._calc_mean_list()
        if self._rangeX_set:
            self._calc_rangeX()
        if self._autoplotdir is not None and self._position_list_set:
            if os.path.isdir(self._autoplotdir):
                if not self._rangeX_set:
                    ranges = [20]
                else:
                    ranges = self._rangeX_set
                for range in ranges:
                    name = self.name() + "_Zoom" + str(range) + ".png"
                    path = os.path.join(self._autoplotdir, name)
                    self.plot_position_list(path, zoom=range, show=False)

    def _check_initialization(self):
        if self._exec is None or self._exec.comparison_matrix is None or self._exec.ranking_matrix is None:
            raise InitializationError
        if self._position_list_set is None:  # Valid for the moment
            raise InitializationError

    def _calc_position_list(self):
        position_list = []
        for elemp, rank_list in enumerate(self._exec.ranking_matrix):
            partial_list = []
            for column, elemg in enumerate(rank_list):
                namep = self._exec.dataset.probe.files[elemp]
                nameg = self._exec.dataset.gallery.files[elemg]
                if self._exec.dataset.same_individual(namep, nameg):
                    partial_list.append(column)
            position_list.append(partial_list)
        self.position_list = np.asarray(position_list, np.uint16)

    def _calc_mean_list(self):
        elem_len, _ = self.position_list.shape
        # self.mean_list = np.zeros((1, ids_len), np.float16)
        # print "Calculating mean list"
        self.mean_list = np.asarray([sum(column) / float(elem_len - 1) for column in zip(*self.position_list)],
                                    np.float16)

    def _calc_rangeX(self):
        self.rangeX = []  # Reset
        for admissible in self._rangeX_set:
            elem_len, size = self.position_list.shape
            ranges = np.zeros(size, np.float16)
            for i in range(size):
                ranges[i] = len(self.position_list[:, i][self.position_list[:, i] <= admissible]) / float(elem_len)
            ranges *= 100
            self.rangeX.append(ranges)
        self.rangeX = np.asarray(self.rangeX, np.float16)

    def plot_position_list(self, fig_name, zoom=None, show=False):
        bins_rank, num_positions = self.position_list.shape
        colors = itertools.cycle(["blue", "red", "green", "yellow", "orange"])
        for i in range(num_positions):
            plt.hist(self.position_list[:, i], bins=bins_rank, label='Pos ' + str(i), histtype='stepfilled', alpha=.8,
                     color=next(colors), cumulative=True, normed=True)
        plt.yticks(np.arange(0, 1.01, 0.05))
        plt.grid(True)
        plt.title("Ranking Histogram")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        # Put a legend below current axis
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        # Zoomed figure
        if zoom:
            plt.axis([0, zoom, 0, 1])
            plt.xticks(range(0, zoom+1, 2))
        plt.savefig(fig_name, bbox_inches='tight')
        if show:
            plt.show()
        # Clean and close figure
        plt.clf()
        plt.close()