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

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
DROIDCAM = 'Realtime : Phone'
IMAGE_ENHANCEMENT='Image Enhancement'
WEBCAM = 'Realtime : Webcam'
MULTIPROCESSING='Realtime Multiprocessing'
# YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO, DROIDCAM,IMAGE_ENHANCEMENT,WEBCAM,MULTIPROCESSING]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'image1_undectected.png'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'image1_detected.png'
ENHANCEMENT_IMAGE_DEFAULT=IMAGES_DIR/'Image1.jpg'
ENHANCEMENT_IMAGE_DEFAULT_ENHANCED=IMAGES_DIR/'Image_Enhanced.jpg'



# Videos config
VIDEO_DIR = ROOT / 'FLOOD_VIDEOS_FOR_DETECTION'
VIDEO_1_PATH = VIDEO_DIR / 'COMBINED_VIDEO(vid_1).mp4'
VIDEO_2_PATH = VIDEO_DIR / 'COMBINED_VIDEO_3(vid_2).mp4'
VIDEO_3_PATH = VIDEO_DIR / 'COMBINED_VIDEO_2(vid_3).mp4'
VIDEO_4_PATH = VIDEO_DIR / 'COMBINED_VIDEO_1(vid_4).mp4'
VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,
    'video_2': VIDEO_2_PATH,
    'video_3': VIDEO_3_PATH,
    'video_4': VIDEO_4_PATH,

}

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolov8l.pt'
# In case of your custome model comment out the line above and
# Place your custom model pt file name at the line below 
# DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'

SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'

# Webcam
WEBCAM_PATH = 0
