from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube
import sys
import cvzone
import math
import settings
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import streamlit_ext as ste


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    # print(res[0])
    res_plotted = res[0].plot()
    
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    # vid_cap = cv2.VideoCapture(source_rtsp)
                    # time.sleep(0.1)
                    # continue
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))


def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker,
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())
    # source_vid = r"D:\Streamlit\videoplayback.mp4"


    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    print(st.sidebar.button('Detect Video Objects'))

    if True:
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


# -------------------------------------------------------------------------------------------------------------------------------------------
#new work here
            
def all_detection(vid_source,run):
    # vid_source = st.sidebar.selectbox(
    #     "Choose a video...", settings.VIDEOS_DICT.keys())
    # # source_vid = r"D:\Streamlit\videoplayback.mp4"

    # with open(settings.VIDEOS_DICT.get(vid_source), 'rb') as video_file:
    #     video_bytes = video_file.read()
    # if video_bytes:
    #     st.video(video_bytes)
    cap=cv2.VideoCapture(vid_source)
    # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # current_frame = st.slider("Select Frame", 0, total_frames - 1, 0)

    # speed_factor = st.slider("Fast Forward Speed", 1, 10, 1)

        # Set video capture properties for fast-forwarding
    # cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame * speed_factor)

    # if (cap.isOpened() == False):
    #     print("!!! Failed cap.isOpened()")
    #     sys.exit(-1)
    # st.sidebar.button("Detect Video Objects")
    # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    # cap.set(3,height) #width  640 1280
    # cap.set(4,width) #height 480 720

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    st_frame=st.empty()

    model=YOLO(r'../yolo_weights/yolov8l.pt')

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"] #len = 80 categories
    
    run1=st.sidebar.button("Screenshot")

    while(True):
     
        success, img=cap.read()
        if success == False:
            print("Video over")
            break

        results=model(img,stream=True) #stream =True makes use of generators and hence is more efficient
        # res_plotted=results[0].plot()
        
     
        if run1: 
            # take_screenshot(img)
            run1=False
            # image = Image.new("RGB", (300, 200), color="white")
            # image.save("output_image.png")
            cv2.imwrite(fr'D:\Learning only\yolo\Snapshot\frame.png', img)
            # st.plotly_chart(img)
            enhancement("iii")
            print("yes")
            print("no")

        # if st.sidebar.button("Screenshot"):
        #         # Save the frame as an image
        #         output_path = "screenshot.png"
        #         take_screenshot(output_path, img)


        if run:
            for i in results:
                boundingboxes=i.boxes
                for j in boundingboxes:
                    # open cv method or cv2 method
                    '''
                    x1,y1,x2,y2=j.xyxy[0]  #or x1,y1,x2,y2=j.xywh
                    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                    print(x1,y1,x2,y2)
                    cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
                    '''

                    #cvzone method- more fancier bboxes
                    x1,y1,x2,y2=j.xyxy[0]
                    x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                    w,h=x2-x1,y2-y1


                    confidence=math.ceil((j.conf[0]*100))/100
                    print(confidence)

                    category=int(j.cls[0])
                    print(category)

                    # if (confidence>0.55 and classNames[category]=='truck') :
                    if(classNames[category]=='person' or classNames[category]=='car'):
                        cvzone.cornerRect(img,(x1,y1,w,h),colorC=(255,0,200))
                        cvzone.putTextRect(img,f'{classNames[category]} {confidence}',(max(0,x1),max(30,y1)),scale=1,thickness=1)
            imS = cv2.resize(img, (960, 540))

            # check if 'p' was pressed and wait for a 'b' press
            key = cv2.waitKey(int(frame_count/1000))
            if (key & 0xFF == ord('p')):

                # sleep here until a valid key is pressed
                while (True):
                    key = cv2.waitKey(0)

                    # check if 'p' is pressed and resume playing
                    if (key & 0xFF == ord('p')):
                        break

                    # check if 'b' is pressed and rewind video to the previous frame, but do not play
                    if (key & 0xFF == ord('b')):
                        cur_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        print('* At frame #' + str(cur_frame_number))

                        prev_frame = cur_frame_number
                        if (cur_frame_number > 1):
                            prev_frame -= 1

                        print('* Rewind to frame #' + str(prev_frame))
                        cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)

                    # check if 'r' is pressed and rewind video to frame 0, then resume playing
                    if (key & 0xFF == ord('r')):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        break

            elif (key & 0xFF == ord('k')):
                        cur_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_number+500)

            elif (key & 0xFF == ord('z')):
                        cur_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame_number-200)

            # exit when 'q' is pressed to quit
            elif (key & 0xFF == ord('q')):
                break
                


            elif (key & 0xFF == ord('s')):
                cur_frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
                cv2.imwrite(fr'D:\Learning only\yolo\Snapshot\frame{cur_frame_number}.png', img)


            # cv2.imshow('image',imS)
            # cv2.imshow('image',imS)
            st_frame.image(imS,
                    caption='Detected Video',
                    channels="BGR",
                    use_column_width=True
                    )

        cv2.waitKey(1)


#-------------------Saving Picture----------------------------
def take_screenshot(image):
    # cv2.imwrite(fr'D:\Learning only\yolo\Snapshot\frame.png', image)    
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a PIL Image
    pil_image = Image.fromarray(frame_rgb)

    # Display the PIL Image in Streamlit
    st.image(pil_image, caption="Object Detection", use_column_width=True)

    # Save the frame as an image
    cv2.imwrite(fr'D:\Learning only\yolo\Snapshot\frame.png', image)    


def enhancement(source_img):
    # image = cv2.imread(source_img)
    # print("hello")
    source_img=np.array(source_img)


    denoised_image = cv2.fastNlMeansDenoisingColored(source_img, None, 	5,10, 7,21)
# cv2.imwrite(r'D:\Learning only\Trial\enhancedone1.png', denoised_image)

    contrast_stretched_image = cv2.normalize(denoised_image, None, 255,0, cv2.NORM_MINMAX, cv2.CV_8UC1)
# cv2.imwrite(r'D:\Learning only\Trial\enhancedone2.png', denoised_image)

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
    sharpened_image = cv2.filter2D(contrast_stretched_image, -1, kernel=kernel) 
    brightness_image = cv2.convertScaleAbs(sharpened_image, alpha=1, beta=5)
    gamma =1.5
    lookup_table = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected_image = cv2.LUT(brightness_image, lookup_table)
    cv2.imwrite(r'enhancedone.png', gamma_corrected_image)
    return gamma_corrected_image




def play_stored_video_1(source_vid):
    model=YOLO(r'../yolo_weights/yolov8l.pt')

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"] #len = 80 categories
    
    # run1=st.sidebar.button("Screenshot")

    if True:
        try:
            vid_cap = cv2.VideoCapture(source_vid)
            st_frame = st.empty()
            # if run1:
            #                 print("yes")
            #                 cv2.imwrite('abc.png',img)
            
            while (vid_cap.isOpened()):
                success, img = vid_cap.read()
                if success:
                        # st.sidebar.button(f'Button {i}', key=f'button_{i}')
                        
                        results=model(img,stream=True) #stream =True makes use of generators and hence is more efficient
                        for i in results:
                            boundingboxes=i.boxes
                            for j in boundingboxes:
                        # open cv method or cv2 method
                                '''
                                x1,y1,x2,y2=j.xyxy[0]  #or x1,y1,x2,y2=j.xywh
                                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                                print(x1,y1,x2,y2)
                                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
                                '''

                                #cvzone method- more fancier bboxes
                                x1,y1,x2,y2=j.xyxy[0]
                                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                                w,h=x2-x1,y2-y1


                                confidence=math.ceil((j.conf[0]*100))/100
                                print(confidence)

                                category=int(j.cls[0])
                                print(category)

                                # if (confidence>0.55 and classNames[category]=='truck') :
                                if(classNames[category]=='person' or classNames[category]=='car' or classNames[category]=='truck'):
                                    cvzone.cornerRect(img,(x1,y1,w,h),colorC=(255,0,200))
                                    cvzone.putTextRect(img,f'{classNames[category]} {confidence}',(max(0,x1),max(30,y1)),scale=1,thickness=1)
                        imS = cv2.resize(img, (960, 540))

                        st_frame.image(imS,
                                       caption='Detected Video',
                                       channels="BGR",
                                       use_column_width=True)
                        cv2.waitKey(1)
                        
                        # i=i+1
                else:
                    vid_cap.release()
                    break
                
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))



import PIL.Image as Image
import skimage.io as io
import numpy as np
import time
from gf import guided_filter
# from numba import jit
import matplotlib.pyplot as plt

class HazeRemoval(object):
    def __init__(self, omega=0.95, t0=0.1, radius=7, r=20, eps=0.001):
        pass

    def open_image(self, img_path):
        img = Image.open(img_path)
        self.src = np.array(img).astype(np.double)/255.
        # self.gray = np.array(img.convert('L'))
        self.rows, self.cols, _ = self.src.shape
        self.dark = np.zeros((self.rows, self.cols), dtype=np.double)
        self.Alight = np.zeros((3), dtype=np.double)
        self.tran = np.zeros((self.rows, self.cols), dtype=np.double)
        self.dst = np.zeros_like(self.src, dtype=np.double)
        

    # @jit
    def get_dark_channel(self, radius=7):
        print("Starting to compute dark channel prior...")
        start = time.time()
        tmp = self.src.min(axis=2)
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0,i-radius)
                rmax = min(i+radius,self.rows-1)
                cmin = max(0,j-radius)
                cmax = min(j+radius,self.cols-1)
                self.dark[i,j] = tmp[rmin:rmax+1,cmin:cmax+1].min()
        print("time:",time.time()-start)

    def get_air_light(self):
        print("Starting to compute air light prior...")
        start = time.time()
        flat = self.dark.flatten()
        flat.sort()
        num = int(self.rows*self.cols*0.001)
        threshold = flat[-num]
        tmp = self.src[self.dark>=threshold]
        tmp.sort(axis=0)
        self.Alight = tmp[-num:,:].mean(axis=0)
        # print(self.Alight)
        print("time:",time.time()-start)

    # @jit
    def get_transmission(self, radius=7, omega=0.90):
        print("Starting to compute transmission...")
        start = time.time()
        for i in range(self.rows):
            for j in range(self.cols):
                rmin = max(0,i-radius)
                rmax = min(i+radius,self.rows-1)
                cmin = max(0,j-radius)
                cmax = min(j+radius,self.cols-1)
                pixel = (self.src[rmin:rmax+1,cmin:cmax+1]/self.Alight).min()
                self.tran[i,j] = 1. - omega * pixel
        print("time:",time.time()-start)

    def guided_filter(self, r=60, eps=0.001):
        print("Starting to compute guided filter trainsmission...")
        start = time.time()
        self.gtran = guided_filter(self.src, self.tran, r, eps)
        print("time:",time.time()-start)

    def recover(self, t0=0.1):
        print("Starting recovering...")
        start = time.time()
        self.gtran[self.gtran<t0] = t0
        t = self.gtran.reshape(*self.gtran.shape,1).repeat(3,axis=2)
        # import ipdb; ipdb.set_trace()
        self.dst = (self.src.astype(np.double) - self.Alight)/t + self.Alight
        self.dst *= 255
        self.dst[self.dst>255] = 255
        self.dst[self.dst<0] = 0
        self.dst = self.dst.astype(np.uint8)
        print("time:",time.time()-start)

    def show(self):
        import cv2
        cv2.imwrite("captured_images/src.jpg", (self.src*255).astype(np.uint8)[:,:,(2,1,0)])
        cv2.imwrite("captured_images/dark.jpg", (self.dark*255).astype(np.uint8))
        cv2.imwrite("captured_images/tran.jpg", (self.tran*255).astype(np.uint8))
        cv2.imwrite("captured_images/gtran.jpg", (self.gtran*255).astype(np.uint8))
        cv2.imwrite("captured_images/dst.jpg", self.dst[:,:,(2,1,0)])
        
        io.imsave("test.jpg", self.dst)
        return self.dst



import threading
from ultralytics import YOLO
import cv2    #use to capture and display images and perform manipulations on them
import cvzone # to display detections- also display fancy rectangle
import math


model=YOLO(r'../yolo_weights/yolov8n.pt')
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"] #len = 80 categories



class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):
    # cv2.namedWindow(previewName)
    cam = cv2.VideoCapture(camID)
    st_frame=st.empty()
    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        print("Camera not present")
        rval = False

    while rval:
        # cv2.imshow(previewName, frame)
        st_frame.image(frame, caption='Detected Video',
                                        channels="BGR",
                                        use_column_width=True
                                       )
        rval, frame = cam.read()
        results1 = model(frame, stream=True)  # stream =True makes use of generators and hence is more efficient

        for i in results1:
            boundingboxes = i.boxes
            for j in boundingboxes:
                # open cv method or cv2 method
                '''
                x1,y1,x2,y2=j.xyxy[0]  #or x1,y1,x2,y2=j.xywh
                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                print(x1,y1,x2,y2)
                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
                '''

                # cvzone method- more fancier bboxes
                x1, y1, x2, y2 = j.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                cvzone.cornerRect(frame, (x1, y1, w, h), colorC=(255, 0, 200))

                confidence = math.ceil((j.conf[0] * 100)) / 100
                print(confidence)

                category = int(j.cls[0])

                cvzone.putTextRect(frame, f'{classNames[category]} {confidence}', (max(0, x1), max(30, y1)), scale=2,
                                   thickness=1)
                
        


        key = cv2.waitKey(1)
        if key == 27:  # exit on ESC
            break
    # cv2.destroyWindow(previewName)

def play_stored_video_realtime(source_vid,source_vid2):
    model=YOLO(r'../yolo_weights/yolov8n.pt')

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"] #len = 80 categories
    
    # run1=st.sidebar.button("Screenshot")
    longitude=10
    latitude=11


    if True:
        try:
            vid_cap = cv2.VideoCapture(source_vid)
            vid_cap1 =cv2.VideoCapture(source_vid2)
            st_frame1 = st.empty()
            st_frame2 = st.empty()
            # if run1:
            #                 print("yes")
            #                 cv2.imwrite('abc.png',img)
            
            while (vid_cap.isOpened() and vid_cap1.isOpened()):
                success, img = vid_cap.read()
                success1,img1 =vid_cap1.read()
                print("both opened")
                if success:
                        # st.sidebar.button(f'Button {i}', key=f'button_{i}')
                        
                        results=model(img,stream=True) #stream =True makes use of generators and hence is more efficient
                        for i in results:
                            boundingboxes=i.boxes
                            for j in boundingboxes:
                        # open cv method or cv2 method
                                '''
                                x1,y1,x2,y2=j.xyxy[0]  #or x1,y1,x2,y2=j.xywh
                                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                                print(x1,y1,x2,y2)
                                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
                                '''

                                #cvzone method- more fancier bboxes
                                x1,y1,x2,y2=j.xyxy[0]
                                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                                w,h=x2-x1,y2-y1


                                confidence=math.ceil((j.conf[0]*100))/100
                                print(confidence)

                                category=int(j.cls[0])
                                print(category)

                                # if (confidence>0.55 and classNames[category]=='truck') :
                                if(classNames[category]=='person' or classNames[category]=='car'):
                                    cvzone.cornerRect(img,(x1,y1,w,h),colorC=(255,0,200))
                                    cvzone.putTextRect(img,f'{classNames[category]} {confidence}',(max(0,x1),max(30,y1)),scale=1,thickness=1)
                        imS = cv2.resize(img, (960, 540))

                        st_frame1.image(imS,
                                       caption=f'Location is {longitude},{latitude}',
                                       channels="BGR",
                                       use_column_width=True)
                        # st.write(f'Location is {longitude},{latitude}')
                        st_frame2.image(img1,
                                       caption=f'Location is {longitude},{latitude}',
                                       channels="BGR",
                                       use_column_width=True)
                        # st.write(f'Location is {longitude},{latitude}')
                        cv2.waitKey(1)
                        
                        # i=i+1
                else:
                    vid_cap.release()
                    break
                
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def play_stored_video_realtime_url_1(source_vid):
    model=YOLO(r'../yolo_weights/yolov8l.pt')

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"] #len = 80 categories
    
    # run1=st.sidebar.button("Screenshot")
    # longitude=10
    # latitude=11

    location_url=source_vid.replace('video','sensors.html')
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    import time
    options = webdriver.ChromeOptions()
    options.headless = True
    # Options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)# use firfox,phantomjs as per requirements

    URL = location_url #url must be valid and live
    print(URL)
    driver.get(URL)
    lat = driver.find_element(By.ID, "lat")
    longi= driver.find_element(By.ID, 'long')
    # print(lat,longi)
    
    # st.sidebar.text(f"Location is {lat.text},{longi.text}")
    # time.sleep(5)
    # st.sidebar.text()


    # print(a)
    if True:
        try:
            vid_cap = cv2.VideoCapture(source_vid)
            # vid_cap1 =cv2.VideoCapture(source_vid2s)
            st_frame1 = st.empty()
            # st_frame2 = st.empty()
            # if run1:
            #                 print("yes")
            #                 cv2.imwrite('abc.png',img)
            
            while (vid_cap.isOpened()):
                success, img = vid_cap.read()
                # success1,img1 =vid_cap1.read()
                print("both opened")
                if success:
                        # st.sidebar.button(f'Button {i}', key=f'button_{i}')
                        
                        results=model(img,stream=True) #stream =True makes use of generators and hence is more efficient
                        for i in results:
                            boundingboxes=i.boxes
                            for j in boundingboxes:
                        # open cv method or cv2 method
                                '''
                                x1,y1,x2,y2=j.xyxy[0]  #or x1,y1,x2,y2=j.xywh
                                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                                print(x1,y1,x2,y2)
                                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
                                '''

                                #cvzone method- more fancier bboxes
                                x1,y1,x2,y2=j.xyxy[0]
                                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                                w,h=x2-x1,y2-y1


                                confidence=math.ceil((j.conf[0]*100))/100
                                print(confidence)

                                category=int(j.cls[0])
                                print(category)

                                # if (confidence>0.55 and classNames[category]=='truck') :
                                if(classNames[category]=='person' or classNames[category]=='car' or classNames[category]=='truck'):
                                    cvzone.cornerRect(img,(x1,y1,w,h),colorC=(255,0,200),t=9)
                                    cvzone.putTextRect(img,f'{classNames[category]} {confidence}',(max(0,x1),max(30,y1)),scale=1,thickness=1)
                        imS = cv2.resize(img, (960, 540))

                        st_frame1.image(imS,
                                       caption=f'Location is {lat.text},{longi.text}',
                                       channels="BGR",
                                       use_column_width=True)
                        # st.write(f'Location is {longitude},{latitude}')
                        # st_frame2.image(img1,
                        #                caption=f'Location is {lat.text},{longi.text}',
                        #                channels="BGR",
                        #                use_column_width=True)
                        # st.write(f'Location is {longitude},{latitude}')
                        cv2.waitKey(1)
                        
                        # i=i+1
                else:
                    vid_cap.release()
                    break
                
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def play_stored_video_realtime_url_2(source_vid,source_vid2):
    model=YOLO(r'../yolo_weights/yolov8l.pt')

    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"] #len = 80 categories
    
    # run1=st.sidebar.button("Screenshot")
    longitude=10
    latitude=11

    location_url=source_vid.replace('video','sensors.html')
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    import time
    options = webdriver.ChromeOptions()
    options.headless = True
    # Options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)# use firfox,phantomjs as per requirements

    URL = location_url #url must be valid and live
    print(URL)
    driver.get(URL)
    lat = driver.find_element(By.ID, "lat")
    longi= driver.find_element(By.ID, 'long')
    # print(lat,longi)
    
    # st.sidebar.text(f"Location is {lat.text},{longi.text}")
    # time.sleep(5)
    # st.sidebar.text()


    # print(a)
    if True:
        try:
            vid_cap = cv2.VideoCapture(source_vid)
            vid_cap1 =cv2.VideoCapture(source_vid2)
            st_frame1 = st.empty()
            st_frame2 = st.empty()
            # if run1:
            #                 print("yes")
            #                 cv2.imwrite('abc.png',img)
            
            while (vid_cap.isOpened() and vid_cap1.isOpened()):
                success, img = vid_cap.read()
                success1,img1 =vid_cap1.read()
                print("both opened")
                if success:
                        # st.sidebar.button(f'Button {i}', key=f'button_{i}')
                        
                        results=model(img,stream=True) #stream =True makes use of generators and hence is more efficient
                        
                        for i in results:
                            boundingboxes=i.boxes
                            for j in boundingboxes:
                        # open cv method or cv2 method
                                '''
                                x1,y1,x2,y2=j.xyxy[0]  #or x1,y1,x2,y2=j.xywh
                                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                                print(x1,y1,x2,y2)
                                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
                                '''

                                #cvzone method- more fancier bboxes
                                x1,y1,x2,y2=j.xyxy[0]
                                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                                w,h=x2-x1,y2-y1


                                confidence=math.ceil((j.conf[0]*100))/100
                                print(confidence)

                                category=int(j.cls[0])
                                print(category)

                                # if (confidence>0.55 and classNames[category]=='truck') :
                                if(classNames[category]=='person' or classNames[category]=='car' or classNames[category]=='truck'):
                                    cvzone.cornerRect(img,(x1,y1,w,h),colorC=(255,0,200))
                                    cvzone.putTextRect(img,f'{classNames[category]} {confidence}',(max(0,x1),max(30,y1)),scale=1,thickness=1)
                        imS = cv2.resize(img, (960, 540))

                        results1=model(img1,stream=True) #stream =True makes use of generators and hence is more efficient
                        
                        for i in results1:
                            boundingboxes=i.boxes
                            for j in boundingboxes:
                        # open cv method or cv2 method
                                '''
                                x1,y1,x2,y2=j.xyxy[0]  #or x1,y1,x2,y2=j.xywh
                                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                                print(x1,y1,x2,y2)
                                cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
                                '''

                                #cvzone method- more fancier bboxes
                                x1,y1,x2,y2=j.xyxy[0]
                                x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
                                w,h=x2-x1,y2-y1


                                confidence=math.ceil((j.conf[0]*100))/100
                                print(confidence)

                                category=int(j.cls[0])
                                print(category)

                                # if (confidence>0.55 and classNames[category]=='truck') :
                                if(classNames[category]=='person' or classNames[category]=='car' or classNames[category]=='truck'):
                                    cvzone.cornerRect(img1,(x1,y1,w,h),colorC=(255,0,200))
                                    cvzone.putTextRect(img1,f'{classNames[category]} {confidence}',(max(0,x1),max(30,y1)),scale=1,thickness=1)
                        imS1 = cv2.resize(img1, (960, 540))

                        



                        st_frame1.image(imS,
                                       caption=f'Location is {lat.text},{longi.text}',
                                       channels="BGR",
                                       use_column_width=True)
                        # st.write(f'Location is {longitude},{latitude}')
                        st_frame2.image(imS1,
                                       caption=f'Location is {lat.text},{longi.text}',
                                       channels="BGR",
                                       use_column_width=True)
                        # st.write(f'Location is {longitude},{latitude}')
                        cv2.waitKey(1)
                        
                        # i=i+1
                else:
                    vid_cap.release()
                    vid_cap1.release()
                    break
                
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


























#--------------------NOT USEFUL RIGHT NOW--------------------------------------------------------------------------------------------
    # def play_youtube_video(conf, model):
    #     """
    # Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    # Parameters:
    #     conf: Confidence of YOLOv8 model.
    #     model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    # Returns:
    #     None

    # Raises:
    #     None
    # """
    # source_youtube = st.sidebar.text_input("YouTube Video url")

    # is_display_tracker, tracker = display_tracker_options()

    # if st.sidebar.button('Detect Objects'):
    #     try:
    #         yt = YouTube(source_youtube)
    #         stream = yt.streams.filter(file_extension="mp4", res=720).first()
    #         vid_cap = cv2.VideoCapture(stream.url)

    #         st_frame = st.empty()
    #         while (vid_cap.isOpened()):
    #             success, image = vid_cap.read()
    #             if success:
    #                 _display_detected_frames(conf,
    #                                          model,
    #                                          st_frame,
    #                                          image,
    #                                          is_display_tracker,
    #                                          tracker,
    #                                          )
    #             else:
    #                 vid_cap.release()
    #                 break
    #     except Exception as e:
    #         st.sidebar.error("Error loading video: " + str(e))
