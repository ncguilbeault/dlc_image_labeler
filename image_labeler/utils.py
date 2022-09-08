import cv2
import numpy as np
import yaml

def get_total_frame_number_from_video(video_path):
    capture = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    total_frame_number = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return total_frame_number

def get_fps_from_video(video_path):
    capture = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    video_fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    return video_fps

def get_frame_size_from_video(video_path):
    capture = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    capture.release()
    return frame_size

def get_video_format_from_video(video_path):
    capture = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    video_format = capture.get(cv2.CAP_PROP_FORMAT)
    capture.release()
    return video_format

def get_video_frame(video_path, frame_number = 0, convert_to_grayscale = True):

    video_n_frames = get_total_frame_number_from_video(video_path)

    if frame_number > video_n_frames:
        frame_number = video_n_frames

    capture = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)

    # Set the frame number to load.
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, original_frame = capture.read()
    frame = None
    if success:
        if convert_to_grayscale:
            frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY).astype(np.uint8).copy()
        else:
            frame = original_frame.astype(np.uint8).copy()

    return success, frame

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

