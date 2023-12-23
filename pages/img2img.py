import streamlit as st
from PIL import Image

import tensorflow as tf
import numpy as np

# from api.img2img import Img2ImgStub

from flare_correction.remove_flare import BlareRemoval

def draw_image(image, text, col):
    col.write(text)
    col.image(image)

def app_handle_image(bytesImage):
    image = Image.open(bytesImage)
    draw_image(image, "Original Image :camera:", col1)
    # transformed_image = Img2ImgStub().process(image)

    ten_in = tf.convert_to_tensor(np.asarray(image))

    model = BlareRemoval()
    out_tup = model.Process(ten_in)
    out = out_tup[2]
    out = np.array(out)*255
    print(out[0, :10])
    print(out.shape)

    out = Image.fromarray(np.clip(out,0,255).astype(np.uint8))

    draw_image(out, "Processed Image :camera:", col2)

col1, col2 = st.columns(2)

uploaded_image = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_image is None:
    uploaded_image = open("static/example_image.jpeg", "rb")

app_handle_image(uploaded_image)

