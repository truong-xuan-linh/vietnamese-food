import ast
import numpy as np
import pandas as pd
import plotly.express as px
import urllib.request
import gdown
import streamlit as st
import streamlit.components.v1 as components
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing import image
import os


classes = [
    'banh_beo',
    'banh_bot_loc', 
    'banh_can', 
    'banh_cuon', 
    'banh_gio', 
    'banh_khot', 
    'banh_mi', 
    'banh_mi_nuong', 
    'banh_trang_nuong', 
    'banh_trang_tron', 
    'banh_xeo', 
    'bap_xao', 
    'bot_chien', 
    'bun_bo_hue', 
    'bun_cha', 
    'bun_dau_mam_tom', 
    'bun_thit_nuong', 
    'cao_lau', 
    'chao_long', 
    'chuoi_chien', 
    'com_chay', 
    'com_tam', 
    'goi_cuon', 
    'mi_quang', 
    'nem_chua', 
    'pha_lau', 
    'pho', 
    'sup_cua', 
    'xoai_lac', 
    'xoi_gac'
]
with open("food.txt", "r", encoding="utf-8") as f:
    food = ast.literal_eval(f.read())

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def plot_probs(outputs):
    probs = pd.Series(np.round(outputs * 100, 2), classes)
    probs = probs.sort_values(ascending=False).reset_index()
    probs.columns = ['Class', 'Probability']
    fig = px.bar(probs, x='Class', y='Probability')
    fig.update_layout(xaxis_tickangle=55)
    st.plotly_chart(fig, use_container_width=True)

st.markdown(
    "<h1 style='text-align: center;'>VIETNAMESE STREET FOOD CLASSIFICATION</h1> ",
    unsafe_allow_html=True
    )

st.markdown(
    '''
    <center>
        <img 
            src='https://nld.mediacdn.vn/2018/12/11/ea73ea12-abc6-4e15-8dc7-da46ed1b1cb4-epathailandmissuniverse201838-1544494397775357447804.jpg' 
            style='width: 95%;'
        >
    </center><br/>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    "<h1 style='text-align: center;'>Input</h1> ",
    unsafe_allow_html=True
    )

uploaded_file = st.file_uploader("Choose a file")
url = st.text_input(
    'Image Url: ', 
    'https://cdn.tgdd.vn/Files/2020/03/09/1241004/3-mon-banh-mi-kep-la-mieng-hap-dan-thom-ngon-kho-cuong-13.jpg'
)
st.write('')
st.write('')

if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    st.image(bytes_data, use_column_width=True)
    with open('./test.jpg', 'wb') as f: 
        f.write(bytes_data)
elif url:
    urllib.request.urlretrieve(url, './test.jpg')
    st.markdown(
        f"<center><img src='{url}' style='width: 95%;'></center>",
        unsafe_allow_html=True
    )

st.markdown(
    "<h1 style='text-align: center;'>Output</h1> ",
    unsafe_allow_html=True
    )

img_test = preprocess_image('./test.jpg')
model_path = 'Model/EfficientNet_B0_v2_Final/EfNet0_fine_tune_model.h5'
if "model" not in st.session_state:
    if "EfNet0_fine_tune_model.h5" not in os.listdir(("Model/EfficientNet_B0_v2_Final")):
        gdown.download("https://drive.google.com/uc?id=1Jw-Ub66Vu7G2Osyc1oOhg3H4ZeqxoZxt", model_path, quiet=False)
    st.session_state.model = load_model(model_path)
pred_probs = st.session_state.model.predict(img_test)[0]

index = np.argmax(pred_probs)
label = classes[index]
audio_path = 'food_audio/' + label + '.mp3'
if (pred_probs[index] * 100) >= 70.0:
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()
    st.audio(audio_bytes, format = 'audio/ogg', start_time = 0)
    
    st.markdown(food[label])
    st.markdown(f"**Probability:** {pred_probs[index] * 100:.2f}%")
else:
    st.markdown(
        "<h1 style='text-align: center;'>Can't identify this food</h1> ",
        unsafe_allow_html=True
    )
    
plot_probs(pred_probs)


