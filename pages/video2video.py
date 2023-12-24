import tempfile
import streamlit as st
from PIL import Image

import cv2
import tensorflow as tf

from flare_correction.remove_flare import BlareRemoval, apply_flare_model
from color_correction.color_correction import wb_autocorrect

from shadow_correction.shadow_models import *
# from api.img2img import Img2ImgStub

import numpy as np
from flare_correction.remove_flare import BlareRemoval
import uuid

def handle_shadow(image):
    if "model_restoration" in st.session_state:
        model_restoration, model_detection = st.session_state["model_restoration"], st.session_state["model_detection"]
    else:
        model_restoration, model_detection = get_shadow_models()
        st.session_state["model_restoration"], st.session_state["model_detection"] = model_restoration, model_detection

    # refine вроде не существенно влияет на качество, но сильно замедляет
    print(image[0,:10])
    small_img = cv2.resize(image, (512,512))
    data = {
        "image": small_img,
        "im_shape": small_img.shape,
        "full_image": small_img
    }
    detector_output = run_detector(model_detection, data, refine=False)
    data = {
        "image": small_img,
        "im_shape": small_img.shape,
        "full_image": small_img
    }
    rgb_restored = run_shadowformer(model_restoration, data, detector_output)
    out = rgb_restored.astype(np.uint8)[..., ::-1]
    out = cv2.resize(out, (image.shape[1], image.shape[0]))
    return out

def draw_video_bytes(video, text, col):
    col.write(text)
    col.video(video, format="video/mp4")

def apply_model(image):
    image = np.asarray(image)
    out = image
    if "feature_color_correction" in st.session_state and st.session_state["feature_color_correction"]:
        out = wb_autocorrect(out[..., ::-1])[..., ::-1]

    if "feature_shadow_correction" in st.session_state and st.session_state["feature_shadow_correction"]:
        out = handle_shadow(out)

    if "feature_flare_correction" in st.session_state and st.session_state["feature_flare_correction"]:
        cached_model = None
        if "model_flare_corr" in st.session_state:
            cached_model = st.session_state["model_flare_corr"]
        out, cached_model = apply_flare_model(out, cached_model)
        st.session_state["model_flare_corr"] = cached_model

    return out


def app_handle_video(bytes_video):
    draw_video_bytes(bytes_video, "Original Video :camera:", col1)
    
    with tempfile.NamedTemporaryFile() as temp:
        temp.write(bytes_video.read())
        video_stream = cv2.VideoCapture(temp.name)

    out_frames = []
    fps = 30 # video_stream.get(cv2.CAP_PROP_FPS)
    count = fps*5
    while count > 0:
        count-=1
        flag, frame = video_stream.read()
        if not flag:
            break

        frame = apply_model(frame)

        out_frames.append(frame)
    if len(out_frames) == 0:
        return
    fps = max(20,int(video_stream.get(cv2.CAP_PROP_FPS)/10*10))
    # st.text(f"FPS: {fps}")
    height, width, layers = out_frames[0].shape
    filename = str(uuid.uuid4()) + '.mp4'
    video = cv2.VideoWriter('/root/xmas/' + filename,cv2.VideoWriter_fourcc(*'MP4V'),int(fps),(width,height))
    _ = [video.write(i) for i in out_frames]
    print("VIDEO", _)
    video.release()
    video_stream.release()

    os.system("ffmpeg -i " + '/root/xmas/' + filename + " -y -vcodec libx264 " + '/root/xmas/f' + filename)

    transformed_video_bytes = open('/root/xmas/f' + filename, "rb")
    print("BYTES", transformed_video_bytes)
    # transformed_video_bytes = bytes_video
    draw_video_bytes(transformed_video_bytes, "Processed Video :camera:", col2)


on = st.toggle('Цветокоррекция')
if on:
    # st.write('Цветокоррекция активирована!')
    st.session_state["feature_color_correction"] = True
else:
    st.session_state["feature_color_correction"] = False

on = st.toggle('Удаление теней')
if on:
    # st.write('Удаление теней активировано!')
    st.session_state["feature_shadow_correction"] = True
else:
    st.session_state["feature_shadow_correction"] = False

on = st.toggle('Удаление бликов')
if on:
    # st.write('Удаление бликов активировано!')
    st.session_state["feature_flare_correction"] = True
else:
    st.session_state["feature_flare_correction"] = False


col1, col2 = st.columns(2)

uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4"])
#if uploaded_video is None:
#    uploaded_file = open("static/demo.mp4", "rb")
#    uploaded_video = uploaded_file

if uploaded_video is not None:
    app_handle_video(uploaded_video)
