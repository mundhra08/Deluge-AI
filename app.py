# Python In-built packages
from pathlib import Path
import PIL
import cv2
import sys
# External packages
import streamlit as st
import numpy as np

# Local Modules
import settings
import helper
import time

# Setting page layout
st.set_page_config(
    page_title="Deluge AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# st.page_link("your_app.py", label="Home", icon="🏠")
# st.page_link("pages/page_1.py", label="Page 1", icon="1️⃣")
# st.page_link("pages/page_2.py", label="Page 2", icon="2️⃣", disabled=True)
# st.page_link(r"/real_time_video_processing.html", label="Google", icon="🌎")

# Main page heading
st.title("Deluge AI")

# Sidebar
st.sidebar.header("Deluge AI")

# Model Options
# model_type = st.sidebar.radio(
#     "Select Task", ['Detection',])
model_type='Detection'
# confidence = float(st.sidebar.slider(
#     "Select Model Confidence", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
# elif model_type == 'Segmentation':
#     model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("SAR Operations")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    # conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    # vid_source = st.sidebar.file_uploader(
    #     "Choose an video...", type=("mp4"))
    # file_name = vid_source.name
    # st.write(f"File Name: {file_name}")
    # print(vid_source)
    # run1=st.sidebar.button("Detect Custom Video Objects")


    vid_source = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())
    # print(settings.VIDEOS_DICT[vid_source])

    vid_main_source=str(settings.VIDEOS_DICT[vid_source])
    # print(vid_main_source)
    # source_vid = r"D:\Streamlit\videoplayback.mp4"
    run=st.sidebar.button("Detect Video Objects")
    with open(settings.VIDEOS_DICT.get(vid_source), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)
    
    cap=cv2.VideoCapture(vid_main_source)

    if (cap.isOpened() == False):
            print("!!! Failed cap.isOpened()")
            sys.exit(-1)
    
    # st.sidebar.button("Detect Video Objects")
    # helper.play_stored_video(0.1, model)    
    # helper.all_detection(vid_main_source,run)
    helper.play_stored_video_1(vid_main_source)    
            


# elif source_radio == settings.RTSP:
#     helper.play_rtsp_stream(confidence, model)

# elif source_radio == settings.YOUTUBE:
#     helper.play_youtube_video(confidence, model)
elif source_radio == settings.IMAGE_ENHANCEMENT:
    source_img = None

    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.ENHANCEMENT_IMAGE_DEFAULT)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.ENHANCEMENT_IMAGE_DEFAULT_ENHANCED)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Enhance'):
                # res = model.predict(uploaded_image,
                #                     conf=confidence
                #                     )
                # uploaded_enhanced_image = PIL.Image.open(r'1000_F_291880492_u42uOLo2sG5RVYeNlth16DLEIRZv9Xat.jpg')
                # enhancedImage=helper.enhancement(uploaded_enhanced_image)
                # source_img=np.array(uploaded_enhanced_image)

                hr=helper.HazeRemoval()
                
                hr.open_image(source_img)
                hr.get_dark_channel()
                hr.get_air_light()
                hr.get_transmission()
                hr.guided_filter()
                hr.recover()
                enhancedImage=hr.show()
                # boxes = res[0].boxes
                # res_plotted = res[0].plot()[:, :, ::-1]
                # cv2.imwrite('ENHANCED_IMAGE.jpeg',enhancedImage)
                st.image(enhancedImage, caption='Detected Image',
                         use_column_width=True)
                # try:
                #     with st.expander("Detection Results"):
                #         for box in boxes:
                #             st.write(box.data)
                #         # pass
                # except Exception as ex:
                #     # st.write(ex)
                #     st.write("No image is uploaded yet!")

elif source_radio == settings.DROIDCAM:
    # st.write("DROIDCAM")
    # ip_address = st.text_input('Ip address')
    # time.sleep(20)
    

    droidcam_ip1 = '192.168.82.110'
    droidcam_port1 = '8080'

    # droidcam_port3 = '4747'
    # droidcam_ip4 = '192.168.137.32'
    # droidcam_port4 = '4747'
    droidcam_ip1

    # Construct the DroidCam URL
    droidcam_url1 = f'http://{droidcam_ip1}:{droidcam_port1}/video'
    # droidcam_url2 = f'http://{droidcam_ip2}:{droidcam_port2}/video'
    # droidcam_url3 = f'http://{droidcam_ip3}:{droidcam_port3}/video'
    # droidcam_url4 = f'http://{droidcam_ip4}:{droidcam_port4}/video'

    helper.play_stored_video_realtime_url_1(droidcam_url1)
    # helper.play_stored_video_realtime_url_2(droidcam_url1,droidcam_url2)

    # helper.play_stored_video_realtime(droidcam_url2)

    # thread1=helper.camThread("Camera 1", droidcam_url1)
    # thread1 = real_time("Camera 1", droidcam_url1)
    # thread2 = real_time("Camera 2", droidcam_url2)
    # thread3 = real_time("Camera 3", droidcam_url3)
    # thread4 = real_time("Camera 4", droidcam_url4)
    

    # thread1.start()
    # thread2.start()
    # thread3.start()
    # thread4.start()
elif source_radio == settings.WEBCAM:
    helper.play_webcam(0.1, model)

elif source_radio == settings.MULTIPROCESSING:
    droidcam_ip1 = '192.168.137.136'
    droidcam_port1 = '8080'
    droidcam_ip2 = '192.168.137.250'
    droidcam_port2 = '8080'

    # ip_address1 = st.text_input('Ip address 1st phone')
    # ip_address2 = st.text_input('Ip address 2nd phone')
    # time.sleep(30)
    # droidcam_ip3 = '192.168.137.248'
    # droidcam_port3 = '4747'
    # droidcam_ip4 = '192.168.137.32'
    # droidcam_port4 = '4747'


    # Construct the DroidCam URL
    # droidcam_ip1=ip_address1
    # droidcam_ip2=ip_address2

    droidcam_url1 = f'http://{droidcam_ip1}:{droidcam_port1}/video'
    droidcam_url2 = f'http://{droidcam_ip2}:{droidcam_port2}/video'
    # droidcam_url3 = f'http://{droidcam_ip3}:{droidcam_port3}/video'
    # droidcam_url4 = f'http://{droidcam_ip4}:{droidcam_port4}/video'

    helper.play_stored_video_realtime_url_2(droidcam_url1,droidcam_url2)

else:
    st.error("Please select a valid source type!")
