from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys
from utils import *

import matplotlib.pyplot as plt

class ImageLabel(QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.image = None
        self.home()

    def home(self):
        self.offset = [0, 0]
        self.scale = 1    

    def set_image(self, image):
        self.image = image
        self.update()

    def paintEvent(self, event):
        if self.image is None or self.image.isNull():
            return
        painter = QPainter(self)
        width = self.width()
        height = self.height()
        imageWidth = self.image.width()
        imageHeight = self.image.height()
        r1 = width / imageWidth
        r2 = height / imageHeight
        self.r = min(r1, r2)
        self.x = (width + self.offset[0] / self.scale * self.r - imageWidth * self.r) / 2
        self.y = (height + self.offset[1] / self.scale * self.r - imageHeight * self.r) / 2
        painter.setTransform(QTransform().translate(self.x, self.y).scale(self.r * self.scale, self.r * self.scale))
        painter.drawImage(QPointF(0,0), self.image)

    def update_params(self, offset, scale):
        self.offset = offset
        self.scale = scale

    # def mousePressEvent(self, event):
    #     print(self.x, self.y)
    #     print('here',event)
    #     super(ImageLabel, self).mousePressEvent(event)

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("DLC Image Labeler")
        self.resize(800, 800)

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        self.menubar = QMenuBar()
        self.menubar.resize(self.size().width(), self.menubar.height())
        self.add_options_to_menubar()
        self.setMenuBar(self.menubar)

        self.main_layout = QGridLayout()
        self.main_widget.setLayout(self.main_layout)
        self.main_layout.setVerticalSpacing(0)
        self.main_layout.setHorizontalSpacing(0)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.gradient = np.linspace(0, 1, 256)
        self.config = None
        self.labelled_frames = {}
        self.frame_number = 0

        self.label_image = ImageLabel()
        self.image = cv2.cvtColor(cv2.imread('test.png', cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2BGR).astype(np.uint8)
        # print(self.image.shape)
        # cv2.imshow('img', self.image)
        self.label_image.set_image(QImage(self.image.data, self.image.shape[1], self.image.shape[0], QImage.Format_RGB888))
        self.scale = 1
        self.offset = np.array([0, 0], dtype = np.float64)
        self.prev_pos = None

        self.frame_window_slider = QSlider(Qt.Horizontal, self.main_widget)
        self.frame_window_slider.resize(self.size())
        self.frame_window_slider.setTickInterval(0)
        self.frame_window_slider.setSingleStep(0)
        self.frame_window_slider.sliderMoved.connect(self.update_frame)
        self.video_path = None

        self.main_layout.addWidget(self.label_image, 0, 0, 1, 0)
        self.main_layout.addWidget(self.frame_window_slider, 1, 0, 1, 0)
        self.show()

    def add_options_to_menubar(self):
        self.options_menu = self.menubar.addMenu('&Options')

        self.open_video_action = QAction('&Open Video', self)
        self.open_video_action.triggered.connect(self.trigger_open_video)
        self.options_menu.addAction(self.open_video_action)

        self.set_dlc_config_action = QAction('&Set DLC Config', self)
        self.set_dlc_config_action.triggered.connect(self.trigger_set_dlc_config)
        self.options_menu.addAction(self.set_dlc_config_action)

        # self.show_text_labels_action = QAction('&Show Text Labels', self)
        # self.show_text_labels_action.triggered.connect(self.trigger_show_text_labels)
        # self.options_menu.addAction(self.show_text_labels_action)

        # self.load_labels_action = QAction('&Load Labels', self)
        # self.load_labels_action.setShortcut('Ctrl+L')
        # self.load_labels_action.setStatusTip('Load Labels')
        # self.load_labels_action.triggered.connect(self.trigger_load_labels)
        # self.options_menu.addAction(self.load_labels_action)

        # self.save_labels_action = QAction('&Save Labels', self)
        # self.save_labels_action.setShortcut('Ctrl+S')
        # self.save_labels_action.setStatusTip('Save Labels')
        # self.save_labels_action.triggered.connect(self.trigger_save_labels)
        # self.options_menu.addAction(self.save_labels_action)

        # self.remove_frame_labels_action = QAction('&Remove Frame Labels', self)
        # self.remove_frame_labels_action.setShortcut('Ctrl+R')
        # self.remove_frame_labels_action.setStatusTip('Remove Frame Labels')
        # self.remove_frame_labels_action.triggered.connect(self.trigger_remove_labels)
        # self.options_menu.addAction(self.remove_frame_labels_action)

        # self.remove_all_labels_action = QAction('&Remove All Labels', self)
        # self.remove_all_labels_action.setShortcut('Ctrl+R')
        # self.remove_all_labels_action.setStatusTip('Remove All Labels')
        # self.remove_all_labels_action.triggered.connect(self.trigger_remove_all_labels)
        # self.options_menu.addAction(self.remove_all_labels_action)

    def trigger_open_video(self):
        self.video_path = QFileDialog.getOpenFileName(self,"Open Video File", "","Video Files (*.avi; *.mp4; *.mov)", options=QFileDialog.Options())[0]
        if self.video_path:
            print(f'Selected video: {self.video_path}')
            success, image = get_video_frame(self.video_path, 0, False)
            if success:
                self.label_image.set_image(QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888))
                self.n_frames = get_total_frame_number_from_video(self.video_path)
                self.frame_window_slider.setMaximum(self.n_frames)
            else:
                print(f'Failed to load frame.')
        else:
            print(f'Failed to load video.')

    def update_frame(self):
        if self.video_path is not None:
            self.frame_number = int(self.frame_window_slider.sliderPosition())
            success, self.image = get_video_frame(self.video_path, self.frame_number, False)
            if success:
                self.label_image.set_image(QImage(self.image.data, self.image.shape[1], self.image.shape[0], QImage.Format_RGB888))

    def wheelEvent(self, event):

        delta = event.angleDelta().y()
        prev_scale = self.scale
        self.scale *= 1.001 ** delta
        delta_scale = self.scale - prev_scale
        # self.offset += np.array([-delta_scale * self.label_image.width() / 2, -delta_scale * self.label_image.height() / 2], dtype = np.float64)
        # self.offset += self.scale * self.offset
        # self.offset +=
        # self.offset = self.offset / self.scale * self.label_image.r / 2 
        self.label_image.update_params(offset = self.offset, scale = self.scale)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.prev_pos is None:
                self.prev_pos = (event.x(), event.y())
        if event.button() == Qt.RightButton:
            if self.config is None:
                self.config = {
                    'bodyparts' : [
                        'test1',
                        'test2'
                    ],
                    'colormap' : 'rainbow',
                    'dotsize' : 3
                }

            cmap = plt.get_cmap(self.config['colormap'])
            labels = self.config['bodyparts']
            if self.frame_number not in self.labelled_frames.keys():
                self.labelled_frames[self.frame_number] = dict([[label, np.array([np.nan, np.nan])] for label in labels])
            all_coords = np.array(list(self.labelled_frames[self.frame_number].values())).astype(float)
            if not np.isnan(all_coords).any():
                self.labelled_frames[self.frame_number] = dict([[label, np.array([np.nan, np.nan])] for label in labels])
                all_coords = np.array(list(self.labelled_frames[self.frame_number].values())).astype(float)
            cmap_range = np.linspace(0, 255, len(labels)).astype(int)
            coords = np.array([(event.pos().x() - self.label_image.x) * (1 / self.label_image.r) * (1 / self.scale), (event.pos().y() - self.label_image.y - self.menubar.height()) * (1 / self.label_image.r) * (1 / self.scale)])
            if (coords < 0).any() or coords[0] > self.image.shape[0] or coords[1] > self.image.shape[1]:
                return
            label_i = np.isnan(all_coords).all(axis=1).argmax()
            self.labelled_frames[self.frame_number][labels[label_i]] = coords
            radius = self.config['dotsize']
            image = self.image.copy()
            for label_i, label in enumerate(self.labelled_frames[self.frame_number].keys()):
                coords = self.labelled_frames[self.frame_number][label]
                if not np.isnan(coords).any():
                    color = [val*255 for val in cmap(cmap_range[label_i])[:3]]
                    image = cv2.circle(image, [int(val) for val in coords], radius, color, -1, cv2.LINE_AA)
            
            self.label_image.set_image(QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888))
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.prev_pos is not None:
                self.prev_pos = None

    def mouseMoveEvent(self, event):

        if event.buttons() == Qt.LeftButton:
            if self.prev_pos is not None:
                delta_pos = ((event.x() - self.prev_pos[0]) / self.width(), (event.y() - self.prev_pos[1]) / self.height())
                self.prev_pos = (event.x(), event.y())
                self.offset += (delta_pos[0] * self.width() * self.scale / self.label_image.r, delta_pos[1] * self.height() * self.scale / self.label_image.r)

            self.label_image.update_params(offset = self.offset, scale = self.scale)
            self.update()

    def trigger_set_dlc_config(self):
        self.dlc_config_file = QFileDialog.getOpenFileName(self,"Open DLC Config File", "","Config Files (*.yaml)", options=QFileDialog.Options())[0]
        if self.dlc_config_file:
            print(f'Config file: {self.dlc_config_file}')
            self.config = load_yaml(self.dlc_config_file)
        else:
            print(f'Failed to load config.')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    app.exec_()