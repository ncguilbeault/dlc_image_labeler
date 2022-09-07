from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import sys
from utils import *
import matplotlib.pyplot as plt
from pathlib import Path
import csv

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
        self.show_text_labels = False
        self.save_directory = Path(__file__).parent.parent.resolve()

        self.label_image = ImageLabel()
        self.image = cv2.cvtColor(cv2.imread('test.png', cv2.IMREAD_UNCHANGED), cv2.COLOR_RGB2BGR).astype(np.uint8)
        # print(self.image.shape)
        # cv2.imshow('img', self.image)
        # self.label_image.set_image(QImage(self.image.data, self.image.shape[1], self.image.shape[0], QImage.Format_RGB888))
        self.update_image()
        self.scale = 1
        self.offset = np.array([0, 0], dtype = np.float64)
        self.prev_pos = None

        self.frame_window_slider = QSlider(Qt.Horizontal, self.main_widget)
        self.frame_window_slider.resize(self.size())
        self.frame_window_slider.setTickInterval(0)
        self.frame_window_slider.setSingleStep(0)
        self.frame_window_slider.sliderMoved.connect(self.update_frame_pos)
        self.video_path = None

        self.main_layout.addWidget(self.label_image, 0, 0, 1, 0)
        self.main_layout.addWidget(self.frame_window_slider, 1, 0, 1, 0)
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
        self.set_dlc_config_action.triggered.connect(self.trigger_set_dlc_config)
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

        # self.remove_all_labels_action = QAction('&Remove All Labels', self)
        # self.remove_all_labels_action.setShortcut('Ctrl+R')
        # self.remove_all_labels_action.setStatusTip('Remove All Labels')
        # self.remove_all_labels_action.triggered.connect(self.trigger_remove_all_labels)
        # self.options_menu.addAction(self.remove_all_labels_action)

    def trigger_open_video(self):
        self.config = None
        self.labelled_frames = {}
        self.video_path = QFileDialog.getOpenFileName(self,"Open Video File", "","Video Files (*.avi; *.mp4; *.mov)", options=QFileDialog.Options())[0]
        if self.video_path:
            print(f'Selected video: {self.video_path}')
            success, self.image = get_video_frame(self.video_path, 0, False)
            if success:
                # self.label_image.set_image(QImage(self.image.data, self.image.shape[1], self.image.shape[0], QImage.Format_RGB888))
                self.n_frames = get_total_frame_number_from_video(self.video_path)
                self.frame_window_slider.setMaximum(self.n_frames - 1)
                self.update_frame_pos()
            else:
                print(f'Failed to load frame.')
        else:
            print(f'Failed to load video.')

    def update_frame_pos(self):
        if self.video_path is not None:
            self.frame_number = int(self.frame_window_slider.sliderPosition())
            success, self.image = get_video_frame(self.video_path, self.frame_number, False)
            if success:
                # self.label_image.set_image(QImage(self.image.data, self.image.shape[1], self.image.shape[0], QImage.Format_RGB888))
                # if self.frame_number not in self.labelled_frames.keys():
                self.update_image()

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

    def update_image(self):
        image = self.image.copy()
        if self.config is not None and self.check_labels_in_frame():
            labels = self.config['bodyparts']
            cmap = plt.get_cmap(self.config['colormap'])
            cmap_range = np.linspace(0, 255, len(labels)).astype(int)
            radius = self.config['dotsize']

            if self.frame_number in self.labelled_frames.keys():
                for label_i, label in enumerate(self.labelled_frames[self.frame_number].keys()):
                    coords = self.labelled_frames[self.frame_number][label]
                    if not np.isnan(coords).any():
                        color = [val*255 for val in cmap(cmap_range[label_i])[:3]]
                        image = cv2.circle(image, [int(val) for val in coords], radius, color, -1, cv2.LINE_AA)
                        if self.show_text_labels:
                            image = cv2.putText(image, label, [int(val) for val in coords], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            
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
                    'scorer': 'nick'
                }

            labels = self.config['bodyparts']
            if self.frame_number not in self.labelled_frames.keys():
                self.labelled_frames[self.frame_number] = dict([[label, np.array([np.nan, np.nan])] for label in labels])

            all_coords = np.array(list(self.labelled_frames[self.frame_number].values())).astype(float)
            if not np.isnan(all_coords).any():
                self.labelled_frames[self.frame_number] = dict([[label, np.array([np.nan, np.nan])] for label in labels])
                all_coords = np.array(list(self.labelled_frames[self.frame_number].values())).astype(float)

            coords = np.array([(event.pos().x() - self.label_image.x) * (1 / self.label_image.r) * (1 / self.scale), (event.pos().y() - self.label_image.y - self.menubar.height()) * (1 / self.label_image.r) * (1 / self.scale)])
            if (coords < 0).any() or coords[0] > self.image.shape[0] or coords[1] > self.image.shape[1]:
                return

            label_i = np.isnan(all_coords).all(axis=1).argmax()
            self.labelled_frames[self.frame_number][labels[label_i]] = coords
            
            self.update_image()

    def check_labels_in_frame(self):
        if self.frame_number not in self.labelled_frames.keys():
            return False
        all_coords = np.array(list(self.labelled_frames[self.frame_number].values())).astype(float)
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

    def trigger_set_dlc_config(self):
        self.config = None
        self.labelled_frames = {}
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
        if self.config is not None and len(self.labelled_frames.keys()) > 0:
            zero_pad_image_name = len(str(max(list(self.labelled_frames.keys()))))
            csv_filename = f'{self.save_directory}\\CollectedData_nick.csv'
            scorer = self.config['scorer']
            labels = self.config['bodyparts']
            with open(csv_filename, 'w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer_data = ['scorer', '', ''] + [scorer] * len(labels) * 2
                writer.writerow(writer_data)
                writer_data = ['bodyparts', '', ''] 
                for label_txt in labels:
                    writer_data += [label_txt] * 2
                writer.writerow(writer_data)
                writer_data = ['coords', '', '']
                for label_txt in labels:
                    writer_data += ['x', 'y']
                writer.writerow(writer_data)
                video_stem = Path(self.video_path).stem
                for labelled_frame_key in self.labelled_frames.keys():
                    success, frame = get_video_frame(self.video_path, labelled_frame_key, False)
                    if success:
                        image_path = f'{self.save_directory}\\img{str(labelled_frame_key).zfill(zero_pad_image_name)}.png'
                        image_name = Path(image_path).name
                        writer_data = ['labeled-data', video_stem, image_name]
                        for label_txt in labels:
                            coord = self.labelled_frames[labelled_frame_key][label_txt]
                            writer_data += list(coord)
                        writer.writerow(writer_data)
                        cv2.imwrite(image_path, frame)
            # print(self.labelled_frames)
            # pass
            print(f'Saved labels: {csv_filename}')
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
        if len(self.labelled_frames.keys()) != 0:
            print(f'Failed to load labels because labels are already loaded. Save existing labels and then remove all labels before loading new ones.')
            return
        self.labelled_frames = {}
        if self.video_path is None:
            print(f'Failed to load labels because no video has been selected.')
            return
        labels_path = QFileDialog.getOpenFileName(self,"Load Labels File", "","Labels Files (*.csv)", options=QFileDialog.Options())[0]
        if labels_path:
            video_stem = Path(self.video_path).stem
            with open(labels_path, 'r') as f:
                data = [row for row in csv.reader(f)]
            if data[3][1] != video_stem:
                print('Failed to load labels because labels are for a different video.')
                return
            print(f'Labels file: {labels_path}')
            new_label_col_idxs = np.arange(3, len(data[0]), 2)
            for frame_data in data[3:]:
                frame_number = int(frame_data[2].split('.')[0][3:])
                for new_label_col_i in new_label_col_idxs:
                    label = data[1][new_label_col_i]
                    x = frame_data[new_label_col_i]
                    y = frame_data[new_label_col_i + 1]
                    self.labelled_frames[frame_number] = {
                        label: np.array([x, y], dtype=float)
                    }
            body_parts = data[1][new_label_col_idxs]
            colormap = 'rainbow'
            dotsize = 1
            scorer = data[0][3]
            if self.config is None:
                self.config = {
                    'bodyparts' : body_parts,
                    'colormap' : colormap,
                    'dotsize' : dotsize,
                    'scorer': scorer
                }
            self.update_frame_pos()
        else:
            print(f'Failed to load video.')

    def trigger_remove_labels(self):
        if self.frame_number in self.labelled_frames.keys():
            del(self.labelled_frames[self.frame_number])
            self.update_image()

    def trigger_remove_all_labels(self):
        self.labelled_frames = {}
        self.update_image()

    def trigger_undo_last_label(self):
        if self.check_labels_in_frame():
            all_coords = np.array(list(self.labelled_frames[self.frame_number].values())).astype(float)
            if not np.isnan(all_coords).all():
                labels = self.config['bodyparts']
                label_i = np.isnan(all_coords).all(axis=1).argmax() - 1
                self.labelled_frames[self.frame_number][labels[label_i]] = np.array([np.nan, np.nan])
        self.update_image()

    def setChildrenFocusPolicy(self, policy):
        def recursiveSetChildFocusPolicy(parentQWidget):
            for childQWidget in parentQWidget.findChildren(QWidget):
                childQWidget.setFocusPolicy(policy)
                recursiveSetChildFocusPolicy(childQWidget)
        recursiveSetChildFocusPolicy(self)

    def keyPressEvent(self, event):
        # print(event.key())
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
        # if event.key() == Qt.Key_Right and self.video_path is not None:
        #     modifiers = QApplication.keyboardModifiers()
        #     if (modifiers & Qt.ShiftModifier):
        #         self.frame_number += 5
        #     else:
        #         self.frame_number += 1
        #     self.frame_window_slider.setSliderPosition(self.frame_number)
        #     self.update_frame_pos()
        
        # print(modifiers)
        # if (modifiers & Qt.ControlModifier) and (modifiers & QtCore.Qt.ShiftModifier)
        # QWidget.keyPressEvent(self, event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    app.exec_()