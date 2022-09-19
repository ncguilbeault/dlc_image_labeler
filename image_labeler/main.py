from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg

import sys
from utils import *
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from pathlib import Path
import csv
import pandas as pd

class ImageLabel(QWidget):
    def __init__(self, parent = None):
        super(ImageLabel, self).__init__(parent)
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
        self.r_width = width / imageWidth
        self.r_height = height / imageHeight
        self.r = min(self.r_width, self.r_height)
        self.x = (width + self.offset[0] / self.scale * self.r - imageWidth * self.r) / 2
        self.y = (height + self.offset[1] / self.scale * self.r - imageHeight * self.r) / 2
        painter.setTransform(QTransform().translate(self.x, self.y).scale(self.r * self.scale, self.r * self.scale))
        painter.drawImage(QPointF(0,0), self.image)

    def update_params(self, offset, scale):
        self.offset = offset
        self.scale = scale

class WindowLevelAdjuster(QMainWindow):

    update_window_level = pyqtSignal(object)

    def __init__(self, image, parent=None):
        super(WindowLevelAdjuster, self).__init__(parent)
        self.setWindowTitle("Window Level Adjustment")
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.float64).swapaxes(0, 1)
        self.image_view = pg.image(self.image, autoRange=True)
        self.image_view.getHistogramWidget().sigLevelsChanged.connect(self.sig_levels_changed)
        self.image_view.setLevels(0, 255)
        self.widget = QWidget()
        self.layout = QGridLayout()
        self.reset_button = QPushButton('Reset Histogram Range')
        self.reset_button.clicked.connect(self.reset_histogram_range)
        self.layout.addWidget(self.image_view, 0, 0)
        self.layout.addWidget(self.reset_button, 1, 0)
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

    def sig_levels_changed(self, hist):
        self.update_window_level.emit(hist.getLevels())

    def reset_histogram_range(self):
        self.image_view.setImage(self.image, autoRange=True, autoLevels=False)
        self.image_view.setLevels(0, 255)
        self.image_view.update()

    def get_processed_image(self, image):
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.float64).swapaxes(0, 1)
        hist = self.image_view.getHistogramWidget()
        hist.disableAutoHistogramRange()
        self.image_view.setImage(self.image, autoRange=True, autoLevels=False)
        self.image_view.getImageItem().render()
        new_image = image = self.image_view.getImageItem().qimage
        return new_image.data

    def keyPressEvent(self, event):
        return self.parent().keyPressEvent(event)

class BodypartLabelWindow(QMainWindow):

    update_bodypart_selected = pyqtSignal(object)

    def __init__(self, parent=None):
        super(BodypartLabelWindow, self).__init__(parent)
        self.setWindowTitle("Bodypart Labels")
        self.resize(300, 300)
        self.checkboxes = []
        self.groupbox = QGroupBox(self)
        self.vertical_layout = QVBoxLayout()
        self.groupbox.setLayout(self.vertical_layout)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.groupbox)
        self.scroll_area.setWidgetResizable(True)
        self.widget = QWidget()
        self.layout = QGridLayout()
        self.layout.addWidget(self.scroll_area, 0, 0)
        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

    def update_groupbox_with_bodyparts(self, bodyparts, colors, individuals = None):
        if len(self.checkboxes) > 0:
            for checkbox in self.checkboxes:
                checkbox.deleteLater()
                del(checkbox)
            self.checkboxes = []            
        if individuals is None:
            for bodypart, color in zip(bodyparts, colors):
                checkbox = QCheckBox(f"{bodypart}")
                hex_color = clrs.to_hex(color)
                checkbox.setStyleSheet(
                    "QCheckBox::indicator:checked"
                            "{"
                            f"background-color : {hex_color};"
                            "border : 3px solid limegreen;"
                            "border-radius :12px;"
                            "}"
                    "QCheckBox::indicator:unchecked"
                            "{"
                            f"background-color : {hex_color};"
                            f"border : 5px solid rgba(255, 255, 255, 0);"
                            "border-radius :12px;"
                            "}"
                )
                self.checkboxes.append(checkbox)
                self.vertical_layout.addWidget(checkbox)
                checkbox.clicked.connect(self.check_bodypart_selection)
        else:
            for individual in individuals:
                for bodypart, color in zip(bodyparts, colors):
                    checkbox = QCheckBox(f"{individual}_{bodypart}")
                    hex_color = clrs.to_hex(color)
                    checkbox.setStyleSheet(
                        "QCheckBox::indicator:checked"
                                "{"
                                f"background-color : {hex_color};"
                                "border : 3px solid limegreen;"
                                "border-radius :12px;"
                                "}"
                        "QCheckBox::indicator:unchecked"
                                "{"
                                f"background-color : {hex_color};"
                                f"border : 5px solid rgba(255, 255, 255, 0);"
                                "border-radius :12px;"
                                "}"
                    )
                    self.checkboxes.append(checkbox)
                    self.vertical_layout.addWidget(checkbox)
                    checkbox.clicked.connect(self.check_bodypart_selection)

    def check_bodypart_selection(self, sender = None):
        if sender is None or isinstance(sender, bool):
            sender = self.sender()
        for i in range(len(self.checkboxes)):
            checkbox = self.checkboxes[i]
            if checkbox != sender and checkbox.isChecked(): 
                checkbox.setChecked(False)
            if checkbox == sender:
                if not checkbox.isChecked():
                    checkbox.setChecked(True)
                else:
                    self.update_bodypart_selected.emit(i)

    def set_bodypart_checked(self, index):
        if index < len(self.checkboxes):
            checkbox = self.checkboxes[index]
            checkbox.setChecked(True)
            self.check_bodypart_selection(sender = checkbox)

    def keyPressEvent(self, event):
        return self.parent().keyPressEvent(event)


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
        self.labeled_frames = {}
        self.frame_number = 0
        self.show_text_labels = False
        self.save_directory = Path(__file__).parent.parent.resolve()
        
        self.image_min = 0
        self.image_max = 255

        self.label_image = ImageLabel(self)
        self.image = cv2.imread('test.png', cv2.IMREAD_REDUCED_COLOR_8).astype(np.uint8)
        self.scale = 1
        self.offset = np.array([0, 0], dtype = np.float64)
        self.prev_pos = None

        self.window_level_adjuster = WindowLevelAdjuster(self.image, self)
        self.window_level_adjuster.update_window_level.connect(self.update_window_level)
        self.update_image()

        self.bodypart_label_window = BodypartLabelWindow(self)
        self.bodypart_label_window.update_bodypart_selected.connect(self.update_bodypart_selected)
        self.bodypart_label_window.show()
        self.bodypart_selected_index = 0

        self.frame_window_slider = QSlider(Qt.Horizontal, self.main_widget)
        self.frame_window_slider.resize(self.size())
        self.frame_window_slider.setTickInterval(0)
        self.frame_window_slider.setSingleStep(0)
        self.frame_window_slider.sliderMoved.connect(self.update_frame_pos)
        self.video_path = None

        self.hbox_widget = QWidget()
        self.hbox_layout = QHBoxLayout()
        self.hbox_widget.setLayout(self.hbox_layout)

        self.frame_number_label = QLabel()
        self.frame_number_label.setAlignment(Qt.AlignCenter)
        self.frame_number_label.setText("Frame number: 0")
        self.hbox_layout.addWidget(self.frame_number_label, Qt.AlignmentFlag.AlignCenter)

        self.prev_frame_labeled_label = QLabel()
        self.prev_frame_labeled_label.setAlignment(Qt.AlignCenter)
        self.prev_frame_labeled_label.setText("Prev labeled frame: None")
        self.hbox_layout.addWidget(self.prev_frame_labeled_label, Qt.AlignmentFlag.AlignCenter)

        self.next_frame_labeled_label = QLabel()
        self.next_frame_labeled_label.setAlignment(Qt.AlignCenter)
        self.next_frame_labeled_label.setText("Next labeled frame: None")
        self.hbox_layout.addWidget(self.next_frame_labeled_label, Qt.AlignmentFlag.AlignCenter)

        self.main_layout.addWidget(self.label_image, 0, 0, 11, 0)
        self.main_layout.addWidget(self.frame_window_slider, 12, 0, 1, 0)
        self.main_layout.addWidget(self.hbox_widget, 13, 0, 1, 0)

        self.setChildrenFocusPolicy(Qt.NoFocus)
        self.show()

    def add_options_to_menubar(self):
        self.options_menu = self.menubar.addMenu('&Options')

        self.open_video_action = QAction('&Open Video', self)
        self.open_video_action.setShortcut('Ctrl+O')
        self.open_video_action.triggered.connect(self.trigger_open_video)
        self.options_menu.addAction(self.open_video_action)

        self.set_dlc_config_action = QAction('&Set DLC Config', self)
        self.set_dlc_config_action.setShortcut('Ctrl+D')
        self.set_dlc_config_action.triggered.connect(self.trigger_load_dlc_config)
        self.options_menu.addAction(self.set_dlc_config_action)

        self.show_text_labels_action = QAction('&Show Text Labels', self)
        self.show_text_labels_action.setShortcut('Ctrl+T')
        self.show_text_labels_action.triggered.connect(self.trigger_show_text_labels)
        self.options_menu.addAction(self.show_text_labels_action)

        self.set_save_directory_action = QAction('&Set Save Directory', self)
        self.set_save_directory_action.setShortcut('Ctrl+P')
        self.set_save_directory_action.triggered.connect(self.trigger_set_save_directory)
        self.options_menu.addAction(self.set_save_directory_action)

        self.load_labels_action = QAction('&Load Labels', self)
        self.load_labels_action.setShortcut('Ctrl+L')
        self.load_labels_action.triggered.connect(self.trigger_load_labels)
        self.options_menu.addAction(self.load_labels_action)

        self.save_labels_action = QAction('&Save Labels', self)
        self.save_labels_action.setShortcut('Ctrl+S')
        self.save_labels_action.triggered.connect(self.trigger_save_labels)
        self.options_menu.addAction(self.save_labels_action)

        self.remove_labels_action = QAction('&Remove Labels', self)
        self.remove_labels_action.setShortcut('Ctrl+R')
        self.remove_labels_action.triggered.connect(self.trigger_remove_labels)
        self.options_menu.addAction(self.remove_labels_action)

        self.remove_all_labels_action = QAction('&Remove All Labels', self)
        self.remove_all_labels_action.setShortcut('Ctrl+Shift+R')
        self.remove_all_labels_action.triggered.connect(self.trigger_remove_all_labels)
        self.options_menu.addAction(self.remove_all_labels_action)

        self.delete_last_label_action = QAction('&Delete Last Label', self)
        self.delete_last_label_action.setShortcut('Ctrl+Z')
        self.delete_last_label_action.triggered.connect(self.trigger_delete_last_label)
        self.options_menu.addAction(self.delete_last_label_action)

        self.adjust_window_level_action = QAction('&Adjust Window Level', self)
        self.adjust_window_level_action.setShortcut('Ctrl+A')
        self.adjust_window_level_action.triggered.connect(self.trigger_show_window_level)
        self.options_menu.addAction(self.adjust_window_level_action)

        self.home_image_action = QAction('&Home Image', self)
        self.home_image_action.setShortcut('Ctrl+H')
        self.home_image_action.triggered.connect(self.trigger_home_image)
        self.options_menu.addAction(self.home_image_action)

        self.show_bodypart_label_window_action = QAction('&Show Bodypart Label Window', self)
        self.show_bodypart_label_window_action.setShortcut('Ctrl+B')
        self.show_bodypart_label_window_action.triggered.connect(self.trigger_show_bodypart_label_window)
        self.options_menu.addAction(self.show_bodypart_label_window_action)

        self.delete_selected_label_action = QAction('&Delete Bodypart Label', self)
        self.delete_selected_label_action.setShortcut('Del')
        self.delete_selected_label_action.triggered.connect(self.trigger_delete_selected_bodypart_label)
        self.options_menu.addAction(self.delete_selected_label_action)

    def trigger_open_video(self):
        self.labeled_frames = {}
        self.video_path = QFileDialog.getOpenFileName(self,"Open Video File", "","Video Files (*.avi; *.mp4; *.mov)", options=QFileDialog.Options())[0]
        if self.video_path != '':
            print(f'Selected video: {self.video_path}')
            success, self.image = get_video_frame(self.video_path, 0, False)
            if success:
                self.n_frames = get_total_frame_number_from_video(self.video_path)
                self.frame_window_slider.setMaximum(self.n_frames - 1)
                self.window_level_adjuster.reset_histogram_range()
                self.update_frame_pos()
            else:
                print(f'Failed to load frame.')
        else:
            self.video_path = None
            print(f'Failed to load video.')

    def update_frame_pos(self):
        if self.video_path is not None:
            self.frame_number = int(self.frame_window_slider.sliderPosition())
            self.frame_number_label.setText(f"Frame number: {self.frame_number}")
            prev_labeled_frame = None
            next_labeled_frame = None
            if len(self.labeled_frames.keys()) > 0:
                all_labeled_frames = sorted([int(frame) for frame in self.labeled_frames.keys()])
                diff_labeled_frames = np.array([np.abs(self.frame_number - frame) for frame in all_labeled_frames])
                frame_idx = diff_labeled_frames.argmin() + 1
                all_labeled_frames = [None] + all_labeled_frames + [None]
                if all_labeled_frames[frame_idx] == self.frame_number:
                    prev_labeled_frame = all_labeled_frames[frame_idx - 1]
                    next_labeled_frame = all_labeled_frames[frame_idx + 1]
                elif all_labeled_frames[frame_idx] > self.frame_number:
                    prev_labeled_frame = all_labeled_frames[frame_idx - 1]
                    next_labeled_frame = all_labeled_frames[frame_idx]
                else:
                    prev_labeled_frame = all_labeled_frames[frame_idx]
                    next_labeled_frame = all_labeled_frames[frame_idx + 1]
            self.prev_frame_labeled_label.setText(f"Prev labeled frame: {prev_labeled_frame}")
            self.next_frame_labeled_label.setText(f"Next labeled frame: {next_labeled_frame}")
            self.bodypart_selected_index = self.get_next_label()
            self.bodypart_label_window.set_bodypart_checked(self.bodypart_selected_index)
            success, self.image = get_video_frame(self.video_path, self.frame_number, False)
            if success:
                self.update_image()

    def wheelEvent(self, event):

        delta = event.angleDelta().y()
        prev_scale = self.scale
        self.scale *= 1.001 ** delta
        delta_scale = self.scale - prev_scale
        self.label_image.update_params(offset = self.offset, scale = self.scale)
        self.update()

    def update_image(self):
        image = self.image.copy()

        image = self.adjust_image_with_window_level(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.config is not None and self.check_labels_in_frame():
            if not self.config['multianimalproject']:
                bodyparts = self.config['bodyparts']
                cmap = plt.get_cmap(self.config['colormap'])
                cmap_range = np.linspace(0, 255, len(bodyparts)).astype(int)
                radius = self.config['dotsize']

                if self.frame_number in self.labeled_frames.keys():
                    for bodypart_i, bodypart in enumerate(self.labeled_frames[self.frame_number].keys()):
                        coords = self.labeled_frames[self.frame_number][bodypart]
                        if not np.isnan(coords).any():
                            color = [val*255 for val in cmap(cmap_range[bodypart_i])[:3]]
                            image = cv2.circle(image, [int(val) for val in coords], radius, color, -1, cv2.LINE_AA)
                            if self.show_text_labels:
                                image = cv2.putText(image, bodypart, [int(val) for val in coords], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            else:
                bodyparts = self.config['multianimalbodyparts']
                cmap = plt.get_cmap(self.config['colormap'])
                cmap_range = np.linspace(0, 255, len(bodyparts)).astype(int)
                radius = self.config['dotsize']

                if self.frame_number in self.labeled_frames.keys():
                    for individual in self.labeled_frames[self.frame_number].keys():
                        for bodypart_i, bodypart in enumerate(self.labeled_frames[self.frame_number][individual].keys()):
                            coords = self.labeled_frames[self.frame_number][individual][bodypart]
                            if not np.isnan(coords).any():
                                color = [val*255 for val in cmap(cmap_range[bodypart_i])[:3]]
                                image = cv2.circle(image, [int(val) for val in coords], radius, color, -1, cv2.LINE_AA)
                                if self.show_text_labels:
                                    image = cv2.putText(image, f"{individual}_{bodypart}", [int(val) for val in coords], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        self.label_image.set_image(QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888))
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
                    'dotsize' : 1,
                    'scorer': 'scorer',
                    'multianimalproject' : False,
                }
                bodyparts = self.config['bodyparts']
                cmap = plt.get_cmap(self.config['colormap'])
                cmap_range = np.linspace(0, 255, len(bodyparts)).astype(int)
                colors = [cmap(cmap_range[i])[:3] for i in range(len(bodyparts))]
                self.bodypart_label_window.update_groupbox_with_bodyparts(bodyparts, colors)

            coords = np.array([(event.pos().x() - self.label_image.x) * (1 / self.label_image.r) * (1 / self.scale), (event.pos().y() - self.label_image.y - self.menubar.height()) * (1 / self.label_image.r) * (1 / self.scale)])
            if (coords < 0).any() or coords[0] > self.image.shape[1] or coords[1] > self.image.shape[0]:
                return
            
            if not self.config['multianimalproject']:

                bodyparts = self.config['bodyparts']

                if self.frame_number not in self.labeled_frames.keys():
                    self.labeled_frames[self.frame_number] = dict([[bodypart, np.array([np.nan, np.nan])] for bodypart in bodyparts])

                self.labeled_frames[self.frame_number][bodyparts[self.bodypart_selected_index]] = coords

                if self.bodypart_selected_index < len(bodyparts) - 1:
                    self.bodypart_selected_index += 1
                else:
                    self.bodypart_selected_index = self.get_next_label()
                
                self.bodypart_label_window.set_bodypart_checked(self.bodypart_selected_index)
                
                self.update_image()

            else:
                bodyparts = self.config['multianimalbodyparts']
                individuals = self.config['individuals']

                if self.frame_number not in self.labeled_frames.keys():
                    self.labeled_frames[self.frame_number] = dict([[individual, dict([[bodypart, np.array([np.nan, np.nan])] for bodypart in bodyparts])] for individual in individuals])
                
                individual_i = int(self.bodypart_selected_index / len(bodyparts))
                bodypart_i = int(self.bodypart_selected_index - (individual_i * len(bodyparts)))
                self.labeled_frames[self.frame_number][individuals[individual_i]][bodyparts[bodypart_i]] = coords

                if self.bodypart_selected_index < (len(bodyparts) * len(individuals)) - 1:
                    self.bodypart_selected_index += 1
                else:
                    self.bodypart_selected_index = self.get_next_label()

                self.bodypart_label_window.set_bodypart_checked(self.bodypart_selected_index)               

                self.update_image()

    def check_labels_in_frame(self):
        if self.frame_number not in self.labeled_frames.keys():
            return False
        if not self.config['multianimalproject']:
            all_coords = np.array(list(self.labeled_frames[self.frame_number].values())).astype(float)
        else:
            all_coords = np.array([self.labeled_frames[self.frame_number][individual][bodypart] for individual, bodyparts in self.labeled_frames[self.frame_number].items() for bodypart in bodyparts]).astype(float)
        if np.isnan(all_coords).all():
            return False
        return True

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.prev_pos is not None:
                self.prev_pos = None

    def mouseMoveEvent(self, event):

        if event.buttons() == Qt.LeftButton:
            if self.prev_pos is not None:
                delta_pos = ((event.x() - self.prev_pos[0]) / self.width(), (event.y() - self.prev_pos[1]) / self.height())
                self.prev_pos = (event.x(), event.y())
                self.offset += (delta_pos[0] * self.width() * self.scale / self.label_image.r_width, delta_pos[1] * self.height() * self.scale / self.label_image.r_height)

            self.label_image.update_params(offset = self.offset, scale = self.scale)
            self.update()

    def trigger_load_dlc_config(self):
        self.config = None
        self.labeled_frames = {}
        self.update_image()
        self.dlc_config_file = QFileDialog.getOpenFileName(self,"Open DLC Config File", "","Config Files (*.yaml)", options=QFileDialog.Options())[0]
        if self.dlc_config_file:
            print(f'Config file: {self.dlc_config_file}')
            self.config = load_yaml(self.dlc_config_file)
            if not self.config['multianimalproject']:
                individuals = None
                bodyparts = self.config['bodyparts']
            else:
                individuals = self.config['individuals']
                bodyparts = self.config['multianimalbodyparts']
            cmap = plt.get_cmap(self.config['colormap'])
            cmap_range = np.linspace(0, 255, len(bodyparts)).astype(int)
            colors = [cmap(cmap_range[i])[:3] for i in range(len(bodyparts))]
            self.bodypart_label_window.update_groupbox_with_bodyparts(bodyparts, colors, individuals = individuals)
            self.bodypart_label_window.set_bodypart_checked(0)
        else:
            print(f'Failed to load config.')

    def trigger_show_text_labels(self):
        if self.config is not None:
            self.show_text_labels = self.show_text_labels == False
            self.update_image()
        else:
            print(f'Failed to show text labels because no config file loaded.')

    def trigger_save_labels(self):
        if self.config is not None and len(self.labeled_frames.keys()) > 0:
            if self.video_path is not None:
                video_stem = Path(self.video_path).stem
            else:
                video_stem = 'test'
            save_stem = Path(self.save_directory).stem
            if video_stem != save_stem:
                message = self.show_message("Video name and save directory are named differently which can lead to problems with DLC. Do you wish to proceed anyways?",
                f"Name of video: {video_stem}\nName of save directory: {save_stem}\n")
                proceed = message.exec()
            else:
                proceed = QMessageBox.Ok
            if proceed != QMessageBox.Ok:
                print('Cancelled saving labels.')
                return
            if "video_sets" in self.config.keys():
                video_crop_params = [self.config["video_sets"][video_key]["crop"] for video_key in self.config["video_sets"].keys() if Path(video_key).stem == video_stem]
                if len(video_crop_params) == 0:
                    print('Failed to load labels because video is not included in video sets of DLC config.')
                    return
                x_offset = int(video_crop_params[0].split(', ')[0])
                y_offset = int(video_crop_params[0].split(', ')[2])
                crop_width = int(video_crop_params[0].split(', ')[1])
                crop_height = int(video_crop_params[0].split(', ')[3])
            else:
                x_offset = 0
                y_offset = 0
            scorer = self.config['scorer']
            if not self.config['multianimalproject']:
                bodyparts = self.config['bodyparts']
                frame_keys = list(self.labeled_frames.keys())
                for frame_key in frame_keys:
                    all_coords = np.array(list(self.labeled_frames[frame_key].values())).astype(float)
                    if np.isnan(all_coords).all():
                        del(self.labeled_frames[frame_key])
                if len(self.labeled_frames.items()) == 0:
                    return
                zero_pad_image_name = len(str(max(list(self.labeled_frames.keys()))))
                columns = pd.MultiIndex.from_product([[scorer], bodyparts, ["x", "y"],], names=["scorer", "bodyparts", "coords",],)
                idx = pd.MultiIndex.from_tuples([("labeled-data", video_stem, f"img{str(labeled_frame_key).zfill(zero_pad_image_name)}.png") for labeled_frame_key in self.labeled_frames.keys()])
                data = np.array([np.array([self.labeled_frames[frame_key][bodypart] - np.array([x_offset, y_offset]) for bodypart in bodyparts]).ravel() for frame_key in self.labeled_frames.keys()], dtype=float)
                df = pd.DataFrame(data, index = idx, columns = columns)
                df.sort_index(inplace = True)
                df.reindex(bodyparts, axis = 1, level = df.columns.names.index('bodyparts'))
                csv_path = f'{self.save_directory}\\CollectedData_{scorer}.csv'
                df.to_csv(csv_path)
                hdf_path = csv_path.split('.csv')[0] + '.h5'
                df.to_hdf(hdf_path, "df_with_missing")
            else:
                bodyparts = self.config['multianimalbodyparts']
                individuals = self.config['individuals']
                frame_keys = list(self.labeled_frames.keys())
                for frame_key in frame_keys:
                    all_coords = np.array([self.labeled_frames[frame_key][individual][bodypart] for individual, bodyparts in self.labeled_frames[frame_key].items() for bodypart in bodyparts]).astype(float)
                    if np.isnan(all_coords).all():
                        del(self.labeled_frames[frame_key])
                if len(self.labeled_frames.items()) == 0:
                    return
                zero_pad_image_name = len(str(max(list(self.labeled_frames.keys()))))
                columns = pd.MultiIndex.from_product([[scorer], individuals, bodyparts, ["x", "y"],], names=["scorer", "individuals", "bodyparts", "coords",],)
                idx = pd.MultiIndex.from_tuples([("labeled-data", video_stem, f"img{str(labeled_frame_key).zfill(zero_pad_image_name)}.png") for labeled_frame_key in self.labeled_frames.keys()])
                data = np.array([np.array([self.labeled_frames[frame_key][individual][bodypart] - np.array([x_offset, y_offset]) for individual in individuals for bodypart in bodyparts]).ravel() for frame_key in self.labeled_frames.keys()], dtype=float)
                df = pd.DataFrame(data, index = idx, columns = columns)
                df.sort_index(inplace = True)
                df.reindex(bodyparts, axis = 1, level = df.columns.names.index('bodyparts'))
                csv_path = f'{self.save_directory}\\CollectedData_{scorer}.csv'
                df.to_csv(csv_path)
                hdf_path = csv_path.split('.csv')[0] + '.h5'
                df.to_hdf(hdf_path, "df_with_missing")
            for labeled_frame_key in self.labeled_frames:
                success, frame = get_video_frame(self.video_path, labeled_frame_key, False)
                if success:
                    image_path = f'{self.save_directory}\\img{str(labeled_frame_key).zfill(zero_pad_image_name)}.png'
                    cv2.imwrite(image_path, frame[y_offset:crop_height, x_offset:crop_width])
            print(f'Saved labels: {csv_path} {hdf_path}')
        else:
            print('Failed to save labels because either no config file loaded or no labeled frames.')

    def trigger_set_save_directory(self):
        self.save_directory = QFileDialog.getExistingDirectory(self,"Select Save Directory")
        if self.save_directory == '':
            self.save_directory = Path(__file__).parent.parent.resolve()
            print(f'Failed to set save directory. Using default directory: {self.save_directory}')
        else:
            print(f'Save directory: {self.save_directory}')

    def trigger_load_labels(self):
        if len(self.labeled_frames.keys()) != 0:
            print(f'Failed to load labels because labels are already loaded. Save existing labels and then remove all labels before loading new ones.')
            return
        self.labeled_frames = {}
        if self.video_path is None:
            print(f'Failed to load labels because no video has been selected.')
            return
        if self.config is None:
            print(f'Failed to load labels because no config file has been selected.')
            return
        labels_path = QFileDialog.getOpenFileName(self,"Load Labels File", "","Labels Files (*.csv)", options=QFileDialog.Options())[0]
        if labels_path and labels_path != '':
            with open(labels_path, 'r') as f:
                data = [row for row in csv.reader(f)]
            video_stem = Path(self.video_path).stem
            if "video_sets" in self.config.keys():
                video_crop_params = [self.config["video_sets"][video_key]["crop"] for video_key in self.config["video_sets"].keys() if Path(video_key).stem == video_stem]
                if len(video_crop_params) == 0:
                    print('Failed to load labels because video is not included in video sets of DLC config.')
                    return
                x_offset = float(video_crop_params[0].split(', ')[0])
                y_offset = float(video_crop_params[0].split(', ')[2])
            else:
                x_offset = 0
                y_offset = 0
            if not self.config['multianimalproject']:
                if data[3][1] != video_stem:
                    print('Failed to load labels because labels are for a different video.')
                    return
                print(f'Labels file: {labels_path}')
                new_bodypart_col_idxs = np.arange(3, len(data[0]), 2)
                for frame_data in data[3:]:
                    frame_number = int(frame_data[2].split('.')[0][3:])
                    self.labeled_frames[frame_number] = {}
                    for new_bodypart_col_i in new_bodypart_col_idxs:
                        bodypart = data[1][new_bodypart_col_i]
                        x = frame_data[new_bodypart_col_i]
                        y = frame_data[new_bodypart_col_i + 1]
                        if x == '' and y == '':
                            x = y = np.nan
                        x = float(x) + x_offset
                        y = float(y) + y_offset
                        self.labeled_frames[frame_number][bodypart] = np.array([x, y], dtype=float)
            else:
                if data[4][1] != video_stem:
                    print('Failed to load labels because labels are for a different video.')
                    return
                print(f'Labels file: {labels_path}')
                new_individual_col_idxs = np.arange(3, len(data[0]), 2 * len(self.config['multianimalbodyparts']))
                new_bodypart_col_idxs = np.arange(3, len(data[0]), 2)
                for i, frame_data in enumerate(data[4:]):
                    frame_number = int(frame_data[2].split('.')[0][3:])
                    self.labeled_frames[frame_number] = {}
                    for new_bodypart_col_i in new_bodypart_col_idxs:
                        if new_bodypart_col_i in new_individual_col_idxs:
                            individual = data[1][new_bodypart_col_i]
                            self.labeled_frames[frame_number][individual] = {}
                        bodypart = data[2][new_bodypart_col_i]
                        x = frame_data[new_bodypart_col_i]
                        y = frame_data[new_bodypart_col_i + 1]
                        if x == '' and y == '':
                            x = y = np.nan
                        x = float(x) + x_offset
                        y = float(y) + y_offset
                        self.labeled_frames[frame_number][individual][bodypart] = np.array([x, y], dtype=float)
           
            self.update_image()
        else:
            print(f'Failed to load labels.')

    def trigger_remove_labels(self):
        if self.frame_number in self.labeled_frames.keys():
            del(self.labeled_frames[self.frame_number])
            self.bodypart_selected_index = 0
            self.bodypart_label_window.set_bodypart_checked(self.bodypart_selected_index)
            self.update_image()

    def trigger_remove_all_labels(self):
        self.labeled_frames = {}
        self.bodypart_selected_index = 0
        self.bodypart_label_window.set_bodypart_checked(self.bodypart_selected_index)
        self.update_image()

    def trigger_delete_last_label(self):
        if self.config is not None and self.check_labels_in_frame():
            label_i = self.get_latest_label()
            if label_i is not None:
                if not self.config['multianimalproject']:
                    bodyparts = self.config['bodyparts']
                    self.labeled_frames[self.frame_number][bodyparts[label_i]] = np.array([np.nan, np.nan])
                else:
                    individuals = self.config['individuals']
                    bodyparts = self.config['multianimalbodyparts']
                    individual_i = int(label_i / len(bodyparts))
                    bodypart_i = int(label_i - (individual_i * len(bodyparts)))
                    self.labeled_frames[self.frame_number][individuals[individual_i]][bodyparts[bodypart_i]] = np.array([np.nan, np.nan])
                label_i = self.get_latest_label()
                if label_i is not None:
                    self.bodypart_selected_index = label_i + 1
                else:
                    self.bodypart_selected_index = 0
                self.bodypart_label_window.set_bodypart_checked(self.bodypart_selected_index)
            self.update_image()

    def setChildrenFocusPolicy(self, policy):
        def recursiveSetChildFocusPolicy(parentQWidget):
            for childQWidget in parentQWidget.findChildren(QWidget):
                childQWidget.setFocusPolicy(policy)
                recursiveSetChildFocusPolicy(childQWidget)
        recursiveSetChildFocusPolicy(self)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.trigger_delete_selected_bodypart_label()
        if self.video_path is not None:
            modifiers = QApplication.keyboardModifiers()
            frame_state_change = False
            if event.key() == Qt.Key_Left:
                frame_state_change = True
                self.frame_number -= 1
                if (modifiers & Qt.ShiftModifier):
                    self.frame_number -= 4
            if event.key() == Qt.Key_Right:
                frame_state_change = True
                self.frame_number += 1
                if (modifiers & Qt.ShiftModifier):
                    self.frame_number += 4
            if frame_state_change:
                self.frame_window_slider.setSliderPosition(self.frame_number)
                self.update_frame_pos()

    def trigger_show_window_level(self):
        self.window_level_adjuster.show()

    def update_window_level(self, window_range):
        window_min, window_max = window_range
        if window_min != self.image_min and window_min >= 0 and window_min <= 255:
            self.image_min = window_min
        if window_max != self.image_max and window_max >= 0 and window_max <= 255:
            self.image_max = window_max
        self.update_image()

    def check_adjustments(self):
        return True if self.image_max != 255 or self.image_min != 0 else False

    def adjust_image_with_window_level(self, image):
        if self.window_level_adjuster is not None:
            image = self.window_level_adjuster.get_processed_image(image)
        return image

    def show_message(self, message, detailed_message = None, title = None):
        message_box = QMessageBox()
        message_box.setText(message)
        if detailed_message is not None:
            message_box.setDetailedText(detailed_message)
        if title is not None:
            message_box.setWindowTitle(title)
        message_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        return message_box

    def trigger_home_image(self):
        self.label_image.home()
        self.update_image()

    def update_bodypart_selected(self, bodypart_selected_i):
        self.bodypart_selected_index = bodypart_selected_i

    def get_latest_label(self):
        latest_label = None
        if self.config is not None and self.frame_number in self.labeled_frames.keys():
            if not self.config['multianimalproject']:
                all_coords = np.array(list(self.labeled_frames[self.frame_number].values())).astype(float)
            else:
                all_coords = np.array([self.labeled_frames[self.frame_number][individual][bodypart] for individual, bodyparts in self.labeled_frames[self.frame_number].items() for bodypart in bodyparts]).astype(float)
            if not np.isnan(all_coords).all():
                if not np.isnan(all_coords).any():
                    latest_label = len(all_coords) - 1
                else:
                    idxs = np.where(~np.isnan(all_coords).all(axis=1))[0]
                    latest_label = idxs[-1]
        return latest_label

    def get_next_label(self):
        next_label = 0
        if self.config is not None and self.frame_number in self.labeled_frames.keys():
            if not self.config['multianimalproject']:
                all_coords = np.array(list(self.labeled_frames[self.frame_number].values())).astype(float)
            else:
                all_coords = np.array([self.labeled_frames[self.frame_number][individual][bodypart] for individual, bodyparts in self.labeled_frames[self.frame_number].items() for bodypart in bodyparts]).astype(float)
            if not np.isnan(all_coords).all():
                if not np.isnan(all_coords).any():
                    next_label = len(all_coords) - 1
                else:
                    idxs = np.where(np.isnan(all_coords).all(axis=1))[0]
                    next_label = idxs[0]
        return next_label

    def trigger_show_bodypart_label_window(self):
        self.bodypart_label_window.show()

    def trigger_delete_selected_bodypart_label(self):
        if self.frame_number in self.labeled_frames.keys():
            if self.config is not None:
                if not self.config['multianimalproject']:
                    bodyparts = self.config['bodyparts']
                    if not np.isnan(self.labeled_frames[self.frame_number][bodyparts[self.bodypart_selected_index]]).all():
                        self.labeled_frames[self.frame_number][bodyparts[self.bodypart_selected_index]] = np.array([np.nan, np.nan])
                        self.update_image()
                else:
                    individuals = self.config['individuals']
                    bodyparts = self.config['multianimalbodyparts']
                    individual_i = int(self.bodypart_selected_index / len(bodyparts))
                    bodypart_i = int(self.bodypart_selected_index - (individual_i * len(bodyparts)))
                    if not np.isnan(self.labeled_frames[self.frame_number][individuals[individual_i]][bodyparts[bodypart_i]]).all():
                        self.labeled_frames[self.frame_number][individuals[individual_i]][bodyparts[bodypart_i]] = np.array([np.nan, np.nan])
                        self.update_image()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    app.exec_()