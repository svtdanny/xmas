import streamlit as st
from PIL import Image

import numpy as np

# from api.img2img import Img2ImgStub

from flare_correction.remove_flare import BlareRemoval, apply_flare_model
from color_correction.color_correction import wb_autocorrect

def draw_image(image, text, col):
    col.write(text)
    col.image(image)

def app_handle_image(bytesImage):
    image_img = Image.open(bytesImage)
    draw_image(image_img, "Original Image :camera:", col1)
    image = np.asarray(image_img)

    # out = apply_flare_model(image)
    out = wb_autocorrect(image)

    out_img = Image.fromarray(out)
    draw_image(out_img, "Processed Image :camera:", col2)

col1, col2 = st.columns(2)

uploaded_image = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_image is None:
    uploaded_image = open("static/example_image.jpeg", "rb")

app_handle_image(uploaded_image)

