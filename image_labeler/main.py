from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import pyqtgraph as pg

import sys
from utils import *
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import pandas as pd

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

class WindowLevelAdjuster(QMainWindow):

    update_window_level = pyqtSignal(object)

    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.image = image.astype(np.float64).swapaxes(0, 1)
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
        self.image = image.astype(np.float64).swapaxes(0, 1)
        hist = self.image_view.getHistogramWidget()
        hist.disableAutoHistogramRange()
        self.image_view.setImage(self.image, autoRange=True, autoLevels=False)
        self.image_view.getImageItem().render()
        new_image = image = self.image_view.getImageItem().qimage
        return new_image.data

class MainWindow(QMainWindow):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("DLC Image Labeler")
        self.resize(800, 800)
        self.window_level_adjuster = None

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
        self.update_image()
        self.scale = 1
        self.offset = np.array([0, 0], dtype = np.float64)
        self.prev_pos = None

        self.window_level_adjuster = WindowLevelAdjuster(self.image, self)
        self.window_level_adjuster.update_window_level.connect(self.update_window_level)

        self.frame_window_slider = QSlider(Qt.Horizontal, self.main_widget)
        self.frame_window_slider.resize(self.size())
        self.frame_window_slider.setTickInterval(0)
        self.frame_window_slider.setSingleStep(0)
        self.frame_window_slider.sliderMoved.connect(self.update_frame_pos)
        self.video_path = None

        self.frame_number_label = QLabel(self)
        self.frame_number_label.setFixedSize(self.menubar.width(), int(self.height()/20))
        self.frame_number_label.setText("Frame number: 0")

        self.main_layout.addWidget(self.label_image, 0, 0, 1, 0)
        self.main_layout.addWidget(self.frame_window_slider, 1, 0, 1, 0)
        self.main_layout.addWidget(self.frame_number_label, 2, 0, 1, 0)

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

        self.undo_last_label = QAction('&Undo Last Label', self)
        self.undo_last_label.setShortcut('Ctrl+Z')
        self.undo_last_label.triggered.connect(self.trigger_undo_last_label)
        self.options_menu.addAction(self.undo_last_label)

        self.adjust_window_level = QAction('&Adjust Window Level', self)
        self.adjust_window_level.setShortcut('Ctrl+A')
        self.adjust_window_level.triggered.connect(self.trigger_show_window_level)
        self.options_menu.addAction(self.adjust_window_level)

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
                                image = cv2.putText(image, label, [int(val) for val in coords], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            else:
                bodyparts = self.config['multianimalbodyparts']
                cmap = plt.get_cmap(self.config['colormap'])
                cmap_range = np.linspace(0, 255, len(bodyparts)).astype(int)
                radius = self.config['dotsize']

                if self.frame_number in self.labeled_frames.keys():
                    for individual in self.labeled_frames[self.frame_number].keys():
                        for label_i, label in enumerate(self.labeled_frames[self.frame_number][individual].keys()):
                            coords = self.labeled_frames[self.frame_number][individual][label]
                            if not np.isnan(coords).any():
                                color = [val*255 for val in cmap(cmap_range[label_i])[:3]]
                                image = cv2.circle(image, [int(val) for val in coords], radius, color, -1, cv2.LINE_AA)
                                if self.show_text_labels:
                                    image = cv2.putText(image, f"{individual}_{label}", [int(val) for val in coords], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
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

            coords = np.array([(event.pos().x() - self.label_image.x) * (1 / self.label_image.r) * (1 / self.scale), (event.pos().y() - self.label_image.y - self.menubar.height()) * (1 / self.label_image.r) * (1 / self.scale)])
            if (coords < 0).any() or coords[0] > self.image.shape[0] or coords[1] > self.image.shape[1]:
                return

            if not self.config['multianimalproject']:

                bodyparts = self.config['bodyparts']

                if self.frame_number not in self.labeled_frames.keys():
                    self.labeled_frames[self.frame_number] = dict([[bodypart, np.array([np.nan, np.nan])] for bodypart in bodyparts])

                all_coords = np.array(list(self.labeled_frames[self.frame_number].values())).astype(float)
                if not np.isnan(all_coords).any():
                    self.labeled_frames[self.frame_number] = dict([[bodypart, np.array([np.nan, np.nan])] for bodypart in bodyparts])
                    all_coords = np.array(list(self.labeled_frames[self.frame_number].values())).astype(float)

                bodypart_i = np.isnan(all_coords).all(axis=1).argmax()
                self.labeled_frames[self.frame_number][bodyparts[bodypart_i]] = coords
                
                self.update_image()
            
            else:
                bodyparts = self.config['multianimalbodyparts']
                individuals = self.config['individuals']

                if self.frame_number not in self.labeled_frames.keys():
                    self.labeled_frames[self.frame_number] = dict([[individual, dict([[bodypart, np.array([np.nan, np.nan])] for bodypart in bodyparts])] for individual in individuals])
                
                all_coords = np.array([self.labeled_frames[self.frame_number][individual][bodypart] for individual, bodyparts in self.labeled_frames[self.frame_number].items() for bodypart in bodyparts]).astype(float)
                if not np.isnan(all_coords).any():
                    self.labeled_frames[self.frame_number] = dict([[individual, dict([[bodypart, np.array([np.nan, np.nan])] for bodypart in bodyparts])] for individual in individuals])
                    all_coords = np.array([self.labeled_frames[self.frame_number][individual][bodypart] for individual, bodyparts in self.labeled_frames[self.frame_number].items() for bodypart in bodyparts]).astype(float)         

                coord_i = np.isnan(all_coords).all(axis=1).argmax()
                individual_i = int(coord_i / len(bodyparts))
                bodypart_i = int(coord_i - (individual_i * len(bodyparts)))
                self.labeled_frames[self.frame_number][individuals[individual_i]][bodyparts[bodypart_i]] = coords

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
                self.offset += (delta_pos[0] * self.width() * self.scale / self.label_image.r, delta_pos[1] * self.height() * self.scale / self.label_image.r)

            self.label_image.update_params(offset = self.offset, scale = self.scale)
            self.update()

    def trigger_load_dlc_config(self):
        self.config = None
        self.labeled_frames = {}
        self.update_frame_pos()
        self.dlc_config_file = QFileDialog.getOpenFileName(self,"Open DLC Config File", "","Config Files (*.yaml)", options=QFileDialog.Options())[0]
        if self.dlc_config_file:
            print(f'Config file: {self.dlc_config_file}')
            self.config = load_yaml(self.dlc_config_file)
        else:
            print(f'Failed to load config.')

    def trigger_show_text_labels(self):
        if self.config is not None:
            self.show_text_labels = self.show_text_labels == False
            self.update_frame_pos()
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
            zero_pad_image_name = len(str(max(list(self.labeled_frames.keys()))))
            scorer = self.config['scorer']
            if not self.config['multianimalprojects']:
                bodyparts = self.config['bodyparts']
                columns = pd.MultiIndex.from_product([[scorer], bodyparts, ["x", "y"],], names=["scorer", "bodyparts", "coords",],)
                idx = pd.MultiIndex.from_tuples([("labeled-data", video_stem, f"img{str(labeled_frame_key).zfill(zero_pad_image_name)}.png") for labeled_frame_key in self.labeled_frames.keys()])
                data = np.array([np.array([self.labeled_frames[frame_key][bodypart] for bodypart in bodyparts]).ravel() for frame_key in self.labeled_frames.keys()], dtype=float)
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
                columns = pd.MultiIndex.from_product([[scorer], individuals, bodyparts, ["x", "y"],], names=["scorer", "individuals", "bodyparts", "coords",],)
                idx = pd.MultiIndex.from_tuples([("labeled-data", video_stem, f"img{str(labeled_frame_key).zfill(zero_pad_image_name)}.png") for labeled_frame_key in self.labeled_frames.keys()])
                data = np.array([np.array([self.labeled_frames[frame_key][individual][bodypart] for individual in individuals for bodypart in bodyparts]).ravel() for frame_key in self.labeled_frames.keys()], dtype=float)
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
                    cv2.imwrite(image_path, frame)
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
            self.update_image()

    def trigger_remove_all_labels(self):
        self.labeled_frames = {}
        self.update_image()

    def trigger_undo_last_label(self):
        if self.config is not None and self.check_labels_in_frame():
            if not self.config['multianimalproject']:
                all_coords = np.array(list(self.labeled_frames[self.frame_number].values())).astype(float)
                if not np.isnan(all_coords).all():
                    bodyparts = self.config['bodyparts']
                    if not np.isnan(all_coords).any():
                        bodypart_i = len(all_coords) - 1
                    else:
                        bodypart_i = np.isnan(all_coords).all(axis=1).argmax() - 1
                    self.labeled_frames[self.frame_number][bodyparts[bodypart_i]] = np.array([np.nan, np.nan])
            else:
                all_coords = np.array([self.labeled_frames[self.frame_number][individual][bodypart] for individual, bodyparts in self.labeled_frames[self.frame_number].items() for bodypart in bodyparts]).astype(float)
                if not np.isnan(all_coords).all():
                    individuals = self.config['individuals']
                    bodyparts = self.config['multianimalbodyparts']
                    if not np.isnan(all_coords).any():
                        coord_i = len(all_coords) - 1
                    else:
                        coord_i = np.isnan(all_coords).all(axis=1).argmax() - 1
                    individual_i = int(coord_i / len(bodyparts))
                    bodypart_i = int(coord_i - (individual_i * len(bodyparts)))
                    self.labeled_frames[self.frame_number][individuals[individual_i]][bodyparts[bodypart_i]] = np.array([np.nan, np.nan])
        # print(self.check_labels_in_frame())
        self.update_image()

    def setChildrenFocusPolicy(self, policy):
        def recursiveSetChildFocusPolicy(parentQWidget):
            for childQWidget in parentQWidget.findChildren(QWidget):
                childQWidget.setFocusPolicy(policy)
                recursiveSetChildFocusPolicy(childQWidget)
        recursiveSetChildFocusPolicy(self)

    def keyPressEvent(self, event):
        if self.video_path is not None:
            modifiers = QApplication.keyboardModifiers()
            state_change = False
            if event.key() == Qt.Key_Left:
                state_change = True
                self.frame_number -= 1
                if (modifiers & Qt.ShiftModifier):
                    self.frame_number -= 4
            if event.key() == Qt.Key_Right:
                state_change = True
                self.frame_number += 1
                if (modifiers & Qt.ShiftModifier):
                    self.frame_number += 4
            if state_change:
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    app.exec_()