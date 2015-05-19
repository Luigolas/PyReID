from copy import copy

__author__ = 'luigolas'

from PyQt4 import QtCore, QtGui, uic
import os
import package.feature_matcher as comparator
import package.preprocessing as preprocessing


# get the directory of this script
path = os.path.dirname(os.path.abspath(__file__))

MainWindowUI, MainWindowBase = uic.loadUiType(
    os.path.join(path, '../gui/mainwindow.ui'))

ImagesSelectionWindowUI, ImagesSelectionWindowBase = uic.loadUiType(
    os.path.join(path, '../gui/imagesselectionwindow.ui'))


class MainWindow(MainWindowBase, MainWindowUI):
    def __init__(self, parent=None):
        MainWindowBase.__init__(self, parent)
        self.setupUi(self)
        self.defaultSetup()

        # Main buttons
        self.ProbeFolderPicker.clicked.connect(self.probe_picker)
        self.GalleryFolderPicker.clicked.connect(self.gallery_picker)
        self.buttonGroupFeMatcherWeights.buttonClicked.connect(self.toggle_weight_user)

        # Preprocess buttons
        self.IluNormAddButton.clicked.connect(self.add_ilunorm)
        self.BTFAddbutton.clicked.connect(self.add_btf)
        self.GrabcutMaskFilePicker.clicked.connect(self.grabcut_mask_file_picker)
        self.GrabcutAddButton.clicked.connect(self.add_grabcut)
        self.MaskFromMatFilePicker.clicked.connect(self.mask_from_mat_file_picker)
        self.MaskFromMatAddButton.clicked.connect(self.add_mask_from_mat)
        self.VerticalregionsAddButton.clicked.connect(self.add_vertical_regions)
        self.SilRegPartAddButton.clicked.connect(self.add_silpart_regions)
        self.GaussianAddButton.clicked.connect(self.add_gaussian)

        self.RemoveSelbutton.clicked.connect(self.removeSel)

        # Run Button
        self.RunButton.clicked.connect(self.run)

        self.preproc = []

    def defaultSetup(self):
        self.ProbeLineEdit.setText(os.path.join(path, '../datasets/viper/cam_a/'))
        self.GalleryLineEdit.setText(os.path.join(path, '../datasets/viper/cam_b/'))
        self.comboBoxComparator.addItems(comparator.method_names)
        self.comboBoxComparator.setCurrentIndex(3)

        # http://stackoverflow.com/a/13661117
        list_model = self.listPreprocess.model()
        list_model.rowsMoved.connect(self.list_preprocess_reorder)

    def list_preprocess_reorder(self, *args):
        print "Layout Changed"
        source = args[1]
        destination = args[-1]
        # http://stackoverflow.com/a/3173159
        self.preproc.insert(destination, self.preproc.pop(source))
        pass

    def probe_picker(self):
        dir_ = QtGui.QFileDialog.getExistingDirectory(None, 'Select a folder:', os.path.join(path, "../datasets/"),
                                                      QtGui.QFileDialog.ShowDirsOnly)
        self.ProbeLineEdit.setText(dir_)

    def gallery_picker(self):
        dir_ = QtGui.QFileDialog.getExistingDirectory(None, 'Select a folder:', os.path.join(path, "../datasets/"),
                                                      QtGui.QFileDialog.ShowDirsOnly)
        self.GalleryLineEdit.setText(dir_)

    def toggle_weight_user(self, RadioButton):
        if RadioButton is self.WeightsUserRadioButton:
            self.WeightsUserLineEdit.setEnabled(True)
        else:
            self.WeightsUserLineEdit.setEnabled(False)

    def add_ilunorm(self):
        from package.image import colorspace_name as cs_name
        colorspace = cs_name.index(self.IluNormComboBox.currentText())
        ilunorm = preprocessing.Illumination_Normalization(colorspace)
        self.add_preproc(ilunorm)

    def add_btf(self):
        btf = preprocessing.BTF(str(self.comboBoxBTF.currentText()))
        self.add_preproc(btf)

    def grabcut_mask_file_picker(self):
        file_ = QtGui.QFileDialog.getOpenFileName(None, 'Select a .txt file:',
                                                  os.path.join(path, "../resources/masks/"))
        self.GrabcutMaskLineEdit.setText(file_)

    def add_grabcut(self):
        try:
            grabcut = preprocessing.Grabcut(str(self.GrabcutMaskLineEdit.text()), self.GrabcutItersSpinBox.value())
            self.add_preproc(grabcut)
        except (ValueError, IOError) as e:
            QtGui.QMessageBox.about(self, "Error", "File name is not valid")

    def mask_from_mat_file_picker(self):
        file_ = QtGui.QFileDialog.getOpenFileName(None, 'Select a .mat file:',
                                                  os.path.join(path, "../resources/masks/"))
        self.MaskFromMatLineEdit.setText(file_)

    def add_mask_from_mat(self):
        try:
            maskfrommat = preprocessing.MasksFromMat(str(self.MaskFromMatLineEdit.text()))
            self.add_preproc(maskfrommat)
        except (ValueError, IOError) as e:
            QtGui.QMessageBox.about(self, "Error", "File name is not valid")

    def add_vertical_regions(self):
        vert = preprocessing.VerticalRegionsPartition()
        self.add_preproc(vert)

    def add_silpart_regions(self):
        alpha = self.SilRegPartAlphaSpinBox.value()
        sub_divisions = self.SilRegPartDivisionSpinBox.value()
        silpart = preprocessing.SilhouetteRegionsPartition(alpha, sub_divisions)
        self.add_preproc(silpart)

    def add_gaussian(self):
        alpha = self.GaussianAlphaSpinBox.value()
        kernel = str(self.GaussianKernelComboBox.currentText())
        sigmas = [self.GaussianSigma1SpinBox.value(), self.GaussianSigma2SpinBox.value()]
        deviations = [self.GaussianDeviation1SpinBox.value(), self.GaussianDeviation2SpinBox.value()]
        gaussian = preprocessing.GaussianMap(alpha, kernel, sigmas, deviations)
        self.add_preproc(gaussian)

    def add_preproc(self, item):
        self.preproc.append(item)
        dict_name = item.dict_name()
        name = dict_name['name']
        if 'params' in dict_name.keys():
            params = dict_name['params']
        else:
            params = ''
        item = QtGui.QListWidgetItem(name + ' ' + params, )
        self.listPreprocess.addItem(item)

    def removeSel(self):
        listItems = self.listPreprocess.selectedItems()
        if not listItems: return
        for item in listItems:
            row = self.listPreprocess.row(item)
            self.listPreprocess.takeItem(row)
            self.preproc.pop(row)

    def run(self):
        from package.app import run_image_selection
        run_image_selection(self.preproc)


class ImagesSelectionForm(ImagesSelectionWindowBase, ImagesSelectionWindowUI):
    def __init__(self, parent=None):
        ImagesSelectionWindowBase.__init__(self, parent)
        self.setupUi(self)
        self.pushButtonNextIteration.clicked.connect(self.next_iteration)
        self.pushButtonMarkSolution.clicked.connect(self.mark_solution)
        self.pushButtonNextProbe.clicked.connect(self.next_probe)
        self.custom_setup()

    def custom_setup(self):
        self.labelSolution.setText("")

    def addImage(self, image_path, enabled=True):
        # self.ui.ImagesContainer.addItem(QtGui.QListWidgetItem(QtGui.QIcon(image_path), "0"))
        img = QtImage(image_path)

        item = QtGui.QListWidgetItem("Selected")
        item.setSizeHint(img.sizeHint())
        img.setEnabled(enabled)

        self.ImagesContainer.addItem(item)
        self.ImagesContainer.setItemWidget(item, img)

    def setProbe(self, probe_path, size=(72, 192)):
        self.labelProbe.setPixmap(QtGui.QPixmap(probe_path))
        self.labelProbe.setFixedSize(QtCore.QSize(*size))
        self.labelProbe.setScaledContents(True)

    def update(self, post_ranker):
        self.labelIterations.setText(str(post_ranker.iteration))
        self.labelTargetPosition.setText(str(post_ranker.target_position))
        self.setProbe(post_ranker.execution.dataset.probe.files_test[post_ranker.subject])
        self.labelProbeName.setText(post_ranker.probe_name)
        self.labelProbeNumber.setText("Probe %d of %d" % (post_ranker.subject, post_ranker.execution.dataset.test_size))
        self.ImagesContainer.clear()
        for elem in post_ranker.rank_list:
            if elem in post_ranker.strong_negatives or elem in post_ranker.weak_negatives:
                enabled = False
            else:
                enabled = True
            self.addImage(post_ranker.execution.dataset.gallery.files_test[elem], enabled)
        if post_ranker.iteration > 0:
            self.labelSolution.hide()
        self.labelErrorMessage.hide()

    def next_iteration(self):
        strong_negatives = []
        weak_negatives = []
        for index in range(self.ImagesContainer.count()):
            # http://www.qtcentre.org/threads/40439-QListWidget-accessing-item-info-%28listWidget-gt-setItemWidget%29
            elem = self.ImagesContainer.itemWidget(self.ImagesContainer.item(index))
            # for rb in elem.selectedRubberBand:
            if len(elem.selectedRubberBand) > 0:
                rb = elem.selectedRubberBand[0]
                if rb.kind == "similar":
                    weak_negatives.append(index)
                else:
                    strong_negatives.append(index)
        if len(strong_negatives) == 0 and len(weak_negatives) == 0:
            self.printError("You need to select at least one weak negative or strong negative")
            return
        from package.app import iterate_images_selection
        iterate_images_selection(strong_negatives, weak_negatives)

    def mark_solution(self):
        selected = None
        for index in range(self.ImagesContainer.count()):
            # http://www.qtcentre.org/threads/40439-QListWidget-accessing-item-info-%28listWidget-gt-setItemWidget%29
            elem = self.ImagesContainer.itemWidget(self.ImagesContainer.item(index))
            # for rb in elem.selectedRubberBand:
            if len(elem.selectedRubberBand) > 0:
                rb = elem.selectedRubberBand[0]
                if rb.kind == "similar":
                    selected = index
                    break
        if selected is not None:
            from package.app import mark_solution
            mark_solution(selected)

    def target_correct(self, solution):
        if solution:
            self.labelSolution.setText("Target correct")
            self.labelSolution.setStyleSheet("QLabel { color : green; }")
            self.labelSolution.show()
        else:
            self.labelSolution.setText("Target Wrong")
            self.labelSolution.setStyleSheet("QLabel { color : red; }")
            self.labelSolution.show()

    @staticmethod
    def next_probe():
        from package.app import new_probe
        new_probe()

    def printError(self, msg):
        self.labelErrorMessage.setText(msg)
        self.labelErrorMessage.setStyleSheet("QLabel { color : red; }")
        self.labelErrorMessage.show()


class NiceRubberBand(QtGui.QRubberBand):
    def __init__(self, QRubberBand_Shape, parent, kind=None):
        QtGui.QRubberBand.__init__(self, QRubberBand_Shape, parent)
        alpha = 50
        self.kind = kind
        if kind == "similar":
            self.color = QtGui.QColor(0, 255, 0, alpha)
        elif kind == "dissimilar":
            self.color = QtGui.QColor(255, 0, 0, alpha)
        else:
            self.color = QtGui.QColor(255, 255, 255, alpha)

    def paintEvent(self, QPaintEvent):
        rect = QPaintEvent.rect()
        painter = QtGui.QPainter(self)
        br = QtGui.QBrush(self.color)
        painter.setBrush(br)
        pen = QtGui.QPen(self.color, 1)
        painter.setPen(pen)
        # painter.setOpacity(0.3)
        painter.drawRect(rect)


class QtImage(QtGui.QLabel):
    def __init__(self, image, parent=None, size=(72, 192)):
        QtGui.QLabel.__init__(self, parent)
        self.setImage(image)
        self.setFixedSize(QtCore.QSize(*size))
        self.setScaledContents(True)
        self.overRubberBand = NiceRubberBand(QtGui.QRubberBand.Rectangle, self)
        self.setMouseTracking(True)
        self.selectedRubberBand = []

    def setImage(self, img):
        self.setPixmap(QtGui.QPixmap(img))

    def mouseMoveEvent(self, QMouseEvent):
        if QtCore.QRect(0, 25, 72, 70).contains(QMouseEvent.pos()):
            self.overRubberBand.setGeometry(QtCore.QRect(0, 25, 72, 70))
            self.overRubberBand.show()
        elif QtCore.QRect(0, 70, 72, 192).contains(QMouseEvent.pos()):
            self.overRubberBand.setGeometry(QtCore.QRect(0, 95, 72, 97))
            self.overRubberBand.show()
        else:
            self.overRubberBand.hide()

    # def enterEvent(self, QEvent):
    # print "Mouse Entered"
    #     return super(QtImage, self).enterEvent(QEvent)

    def leaveEvent(self, QEvent):
        self.overRubberBand.hide()
        return super(QtImage, self).enterEvent(QEvent)

    def mousePressEvent(self, QMouseEvent):
        if self.overRubberBand.isHidden():
            return

        for rubberBand in self.selectedRubberBand:
            if self.overRubberBand.geometry() == rubberBand.geometry():
                rubberBand.hide()
                self.selectedRubberBand.remove(rubberBand)  # hide and remove
                return

        if QMouseEvent.type() == QtCore.QEvent.MouseButtonPress:
            if QMouseEvent.button() == QtCore.Qt.LeftButton:
                kind = "similar"
            elif QMouseEvent.button() == QtCore.Qt.RightButton:
                kind = "dissimilar"
            else:
                return
            rubberBand = NiceRubberBand(QtGui.QRubberBand.Rectangle, self, kind)
            rubberBand.setGeometry(self.overRubberBand.geometry())
            self.selectedRubberBand.append(rubberBand)
            rubberBand.show()