import tempfile
import streamlit as st
from PIL import Image

import cv2
import tensorflow as tf

# from api.img2img import Img2ImgStub

import numpy as np
from flare_correction.remove_flare import BlareRemoval

def draw_video_bytes(video, text, col):
    col.write(text)
    col.video(video, format="video/mp4")

def apply_model(model, image):
    ten_in = tf.convert_to_tensor(image)

    out_tup = model.Process(ten_in)
    out = out_tup[2]
    out = np.array(out)*255
    print(out[0, :10])
    print(out.shape)

    out = np.clip(out,0,255).astype(np.uint8)
    return out


def app_handle_video(bytes_video):
    draw_video_bytes(bytes_video, "Original Video :camera:", col1)

    # model = Img2ImgStub()
    model = BlareRemoval()
    
    with tempfile.NamedTemporaryFile() as temp:
        temp.write(bytes_video.read())
        video_stream = cv2.VideoCapture(temp.name)

    out_frames = []
    
    count = 10000000000
    while count > 0:
        count-=1
        flag, frame = video_stream.read()
        if not flag:
            break

        frame = apply_model(model, frame)

        out_frames.append(frame)
    if len(out_frames) == 0:
        return
    height, width, layers = out_frames[0].shape
    video = cv2.VideoWriter('/home/sivtsovdt/arcadia/ads/pytorch/embedding_model/tmp_video.mp4',cv2.VideoWriter_fourcc(*'MP4V'),20,(width,height))
    _ = [video.write(i) for i in out_frames]
    print("VIDEO", _)
    video.release()
    video_stream.release()

    transformed_video_bytes = open("/home/sivtsovdt/arcadia/ads/pytorch/embedding_model/tmp_video.mp4", "rb")

    # transformed_video_bytes = bytes_video
    draw_video_bytes(transformed_video_bytes, "Processed Video :camera:", col2)

col1, col2 = st.columns(2)

uploaded_video = st.sidebar.file_uploader("Upload a video", type=["mp4"])
# if uploaded_video is None:
#     uploaded_file = open("static/example_video.mp4", "rb")
#     uploaded_video = uploaded_file

if uploaded_video is not None:
    app_handle_video(uploaded_video)
