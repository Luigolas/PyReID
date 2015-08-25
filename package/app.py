__author__ = 'luigolas'

import sys
import ast
from PyQt4 import QtGui
import os
from package import feature_extractor
from package.image import CS_HSV, CS_BGR, CS_IIP
import package.execution as execution
from package.dataset import Dataset
from package.post_ranker import SAA, LabSP, SAL
import package.feature_matcher as FM
from gui.gui import MainWindow, ImagesSelectionForm


DB = None
multiprocessing = True
appMainForm = None
appImagesForm = None
POP = None
# get the directory of this script
path = os.path.dirname(os.path.abspath(__file__))


def run():
    global appMainForm, appImagesForm
    app = QtGui.QApplication(sys.argv)
    appMainForm = MainWindow()
    appImagesForm = ImagesSelectionForm()
    appMainForm.show()
    sys.exit(app.exec_())


def run_image_selection(preproc):
    global appMainForm, appImagesForm, POP

    probe = str(appMainForm.ProbeLineEdit.text())
    gallery = str(appMainForm.GalleryLineEdit.text())

    if appMainForm.comboBoxFeatureExtraction.currentText() == "Histograms":
        if appMainForm.comboBoxColorSpace.currentText() == "HSV":
            colorspace = CS_HSV
        elif appMainForm.comboBoxColorSpace.currentText() == "RGB":
            colorspace = CS_BGR
        else:  # if self.comboBoxColorSpace.currentText == "IIP":
            colorspace = CS_IIP

        if appMainForm.radioButton1D.isChecked():
            dim = "1D"
        else:
            dim = "3D"
        fe = feature_extractor.Histogram(colorspace, ast.literal_eval(str(appMainForm.lineEditBins.text())), dim)
    else:
        fe = None

    if appMainForm.WeightsNoneRadioButton.isChecked():
        weights = None
    elif appMainForm.Weights5RRadioButton.isChecked():
        weights = [0.3, 0.3, 0.15, 0.15, 0.1]
    else:  # if WeightsUserRadioButton.isChecked():
        weights = ast.literal_eval(str(appMainForm.WeightsUserLineEdit.text()))

    Fmatcher = FM.HistogramsCompare(FM.method_names.index(appMainForm.comboBoxComparator.currentText()),
                                        weights)

    train_split = float(appMainForm.lineEditTrainSplit.text())
    if train_split > 1:
        train_split = int(train_split)

    test_split = float(appMainForm.lineEditTestSplit.text())
    if test_split > 1.:
        test_split = int(test_split)

    seed_split = str(appMainForm.lineEditSeed.text())
    if seed_split == "None":
        seed_split = None
    else:
        seed_split = int(seed_split)

    ex = execution.Execution(Dataset(probe, gallery, train_split, test_split, seed_split), preproc, fe, Fmatcher)

    ranking_matrix = ex.run(fe4train_set=True, njobs=1)

    if appMainForm.radioButtonPOPBalancedAndVE.isChecked():
        balanced = True
        visual_expansion_use = True
    elif appMainForm.radioButtonPOPBalanced.isChecked():
        balanced = True
        visual_expansion_use = False
    else:
        balanced = False
        visual_expansion_use = False
    re_score_alpha = appMainForm.ReScoreAlpha.value()
    re_score_method_proportional = appMainForm.radioButtonReScoreProportional.isChecked()

    if appMainForm.radioButtonRegionsNone.isChecked():
        regions = None
    elif appMainForm.radioButtonRegions2R.isChecked():
        regions = [[0], [1]]
    elif appMainForm.radioButtonRegions4R.isChecked():
        regions = [[0, 1], [2, 3]]
    else:  # if appMainForm.radioButtonRegions5R.isChecked():
        regions = [[0, 1], [2, 3, 4]]

    if appMainForm.LabSPradioButton.isChecked():
        POP = LabSP(balanced=balanced, visual_expansion_use=visual_expansion_use, re_score_alpha=re_score_alpha,
                    re_score_method_proportional=re_score_method_proportional, regions=regions)
    else:  # if appMainForm.SAAradioButton.isChecked():
        # POP = SAA(balanced=balanced, visual_expansion_use=visual_expansion_use, re_score_alpha=re_score_alpha,
        #           re_score_method_proportional=re_score_method_proportional, regions=regions)
        POP = SAL(balanced=balanced, visual_expansion_use=visual_expansion_use, re_score_alpha=re_score_alpha,
                  re_score_proportional=re_score_method_proportional, regions=regions)
    POP.set_ex(ex, ranking_matrix)
    appImagesForm.set_regions(regions)

    appImagesForm.update(POP)
    appMainForm.hide()
    appImagesForm.show()


def iterate_images_selection(strong_negatives, weak_negatives):
    global POP, appImagesForm
    POP.new_samples(weak_negatives, strong_negatives)
    msg = POP.iterate()
    if msg == "OK":
        appImagesForm.update(POP)
    else:
        appImagesForm.printError(msg)


def mark_solution(solution):
    global POP, appImagesForm
    if solution == POP.target_position:
        appImagesForm.target_correct(True)
        POP.new_subject()
        appImagesForm.update(POP)
    else:
        appImagesForm.target_correct(False)


def new_probe():
    POP.new_subject()
    appImagesForm.update(POP)




