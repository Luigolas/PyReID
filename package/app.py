__author__ = 'luigolas'

import sys
import ast
from PyQt4 import QtGui
import os
import package.segmenter as segmenter
import package.preprocessing as preprocessing
from package import feature_extractor
from package.image import CS_HSV, CS_BGR, CS_IIP
import package.execution as execution
from package.dataset import Dataset
from package.post_ranker import SAA
import package.comparator as comparator
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


def run_image_selection():
    global appMainForm, appImagesForm, POP

    # Todo set mask for segmenter at GUI
    probe = str(appMainForm.ProbeLineEdit.text())
    gallery = str(appMainForm.GalleryLineEdit.text())

    mask_source = os.path.join(path, "../resources/masks/ViperOptimalMask.txt")
    grabcut = segmenter.Grabcut(mask_source)

    if appMainForm.radioButton6R.isChecked():
        regions = [[[0, 27], [28, 54], [55, 81], [82, 108], [109, 135], [136, 160]], "6R"]
        weights = [0, 0.3, 0.3, 0.15, 0.15, 0.1]
    else:
        regions = [None, None]
        weights = None

    if appMainForm.radioButtonPreprocNone.isChecked():
        preproc = None
    elif appMainForm.radioButtonPreprocCBTF.isChecked():
        preproc = preprocessing.BTF("CBTF")
    else:  # if self.radioButtonPreprocMBTF.ischecked():
        preproc = preprocessing.BTF("ngMBTF")

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
        fe = feature_extractor.Histogram(colorspace, ast.literal_eval(str(appMainForm.lineEditBins.text())), regions[0],
                                         dim, regions[1])
    else:
        fe = None

    comp = comparator.CompHistograms(comparator.method_names.index(appMainForm.comboBoxComparator.currentText()),
        weights)

    train_split = float(appMainForm.lineEditTrainSplit.text())
    if train_split > 1:
        train_split = int(train_split)

    test_split = float(appMainForm.lineEditTestSplit.text())
    if test_split > 1.:
        test_split = int(test_split)

    ex = execution.Execution(Dataset(probe, gallery, train_split, test_split), grabcut, preproc, fe, comp)

    ex.run()

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
    POP = SAA(balanced=balanced, visual_expansion_use=visual_expansion_use, re_score_alpha=re_score_alpha,
              re_score_method_proportional=re_score_method_proportional)
    POP.set_ex(ex)
    appImagesForm.update(POP)
    appMainForm.hide()
    appImagesForm.show()

    # conf.run()
    # conf.to_excel("../results/Salpica.xls")


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




