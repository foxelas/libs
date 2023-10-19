import cv2
import streamlit as st
from libs.pytube import YouTube
from . import processing
from libs.foxutils.utils.core_utils import settings
import av

LIVE_CAM_URL = settings['LIVESTREAM']['live_cam_url']
WEB_CAM_URL = settings['LIVESTREAM']['webcam_path']


def video_frame_callback(frame: av.VideoFrame, labels_placeholder,
                         score_threshold=processing.DEFAULT_CONFIDENCE_THRESHOLD) -> av.VideoFrame:
    frame = frame.to_ndarray(format="bgr24")
    image, detections = processing.object_detection(frame, score_threshold)
    if len(detections) > 0:
        print(f'Detections: {len(detections)}')

    return av.VideoFrame.from_ndarray(image, format="bgr24")


def _display_detected_frames(conf, st_frame, image, labels_placeholder):
    """
    Display the detected objects on a video frame .

    Args:
    - conf (float): Confidence threshold for object detection.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - labels_placeholder (Streamlit object): A Streamlit object to display the detected objects.

    Returns:
    None
    """

    image, detections = processing.object_detection(image, score_threshold=conf)

    st_frame.image(image,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )
    if len(detections) > 0:
        labels_placeholder.table(detections)


def play_youtube_video(conf):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.text_input("YouTube Video url", key='youtube_url')
    st.markdown("Try using https://www.youtube.com/watch?v=oGjwx4EfRqU as a YouTube url.")

    if st.button('Use YouTube', key='use_youtube'):
        print('Using YouTube')
        try:
            stop_btn = st.button('Stop', key='stop')

            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            labels_placeholder = st.empty()

            while (vid_cap.isOpened()) and not (stop_btn):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, st_frame, image, labels_placeholder)

                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_rtsp_stream(conf):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption(f'Example URL: {LIVE_CAM_URL}')
    if st.sidebar.button('Use RTSP'):
        try:
            stop_btn = st.sidebar.button('Stop')

            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            labels_placeholder = st.empty()
            while (vid_cap.isOpened() and (not stop_btn)):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, st_frame, image, labels_placeholder)
                else:
                    vid_cap.release()
                    cv2.destroyAllWindows()
                    # vid_cap = cv2.VideoCapture(source_rtsp)
                    # time.sleep(0.1)
                    # continue
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, source_webcam=WEB_CAM_URL):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Returns:
        None

    Raises:
        None
    """
    if st.sidebar.button('Detect Objects'):
        try:
            stop_btn = st.sidebar.button('Stop')

            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            labels_placeholder = st.empty()
            while (vid_cap.isOpened() and (not stop_btn)):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, st_frame, image, labels_placeholder)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def play_dashcam(conf, source_webcam=WEB_CAM_URL, component_func=None):
    """
    Plays a dashcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Returns:
        None

    Raises:
        None
    """
    st.sidebar.caption(f'Pending setup WebRTC')
    if st.sidebar.button('Use Dashcam'):
        try:
            stop_btn = st.sidebar.button('Stop')
            if component_func is not None:
                component_func()

            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            labels_placeholder = st.empty()
            while (vid_cap.isOpened() and (not stop_btn)):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, st_frame, image, labels_placeholder)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))



from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())
# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'video_1.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'video_2.mp4'
VIDEO_3_PATH = VIDEO_DIR / 'video_3.mp4'
VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,
    'video_2': VIDEO_2_PATH,
    'video_3': VIDEO_3_PATH,
}


def play_stored_video(conf):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", VIDEOS_DICT.keys())

    with open(VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            stop_btn = st.sidebar.button('Stop')

            vid_cap = cv2.VideoCapture(
                str(VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            labels_placeholder = st.empty()
            while (vid_cap.isOpened() and (not stop_btn)):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf, st_frame, image, labels_placeholder)
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
