from copy import copy

__author__ = 'luigolas'

from PyQt4 import QtCore, QtGui, uic
import os
import package.comparator as comparator


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
        self.ProbeFolderPicker.clicked.connect(self.probe_picker)
        self.GalleryFolderPicker.clicked.connect(self.gallery_picker)
        self.RunButton.clicked.connect(self.run)
        # self.custom_setup()

    def defaultSetup(self):
        self.ProbeLineEdit.setText(os.path.join(path, '../datasets/viper/cam_a/'))
        self.GalleryLineEdit.setText(os.path.join(path, '../datasets/viper/cam_b/'))
        self.comboBoxComparator.addItems(comparator.method_names)
        self.comboBoxComparator.setCurrentIndex(3)

    def probe_picker(self):
        dir_ = QtGui.QFileDialog.getExistingDirectory(None, 'Select a folder:', os.path.join(path, "../datasets/"),
                                                      QtGui.QFileDialog.ShowDirsOnly)
        self.ProbeLineEdit.setText(dir_)

    def gallery_picker(self):
        dir_ = QtGui.QFileDialog.getExistingDirectory(None, 'Select a folder:', os.path.join(path, "../datasets/"),
                                                      QtGui.QFileDialog.ShowDirsOnly)
        self.GalleryLineEdit.setText(dir_)

    def run(self):
        from package.app import run_image_selection
        run_image_selection()


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
