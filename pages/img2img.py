import streamlit as st
from PIL import Image

import numpy as np

# from api.img2img import Img2ImgStub

from flare_correction.remove_flare import BlareRemoval, apply_flare_model
from color_correction.color_correction import wb_autocorrect

from shadow_correction.shadow_models import *

def draw_image(image, text, col):
    col.write(text)
    col.image(image)

def handle_shadow(image):
    model_restoration, model_detection = get_shadow_models()
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
    #out = detector_output.astype(np.uint8)
    return out

def app_handle_image(bytesImage):
    image_img = Image.open(bytesImage)
    draw_image(image_img, "Изначальное изображение :camera:", col1)
    image = np.array(image_img)
    out = image
    if "feature_color_correction" in st.session_state and st.session_state["feature_color_correction"]:
        out = wb_autocorrect(out[..., ::-1])[..., ::-1]

    if "feature_shadow_correction" in st.session_state and st.session_state["feature_shadow_correction"]:
        out = handle_shadow(out)

    if "feature_flare_correction" in st.session_state and st.session_state["feature_flare_correction"]:
        out, _ = apply_flare_model(out)

    # # out = handle_shadow(image)
    # # print(out[0,:10])
    # out, _ = apply_flare_model(image)
    # # out = wb_autocorrect(image[..., ::-1])[..., ::-1]

    out_img = Image.fromarray(out)
    draw_image(out_img, "Обработанное изображение :camera:", col2)

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

uploaded_image = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_image is None:
    uploaded_image = open("static/fr0_sensorname_bigelt.png", "rb")

app_handle_image(uploaded_image)

