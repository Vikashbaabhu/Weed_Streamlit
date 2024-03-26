import streamlit as st
import tensorflow as tf
import streamlit as st
import pickle
import matplotlib.pyplot as plt
import os
import tempfile
from tqdm import tqdm
import json
import base64

with open('style.css') as f :
  st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
  
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://st4.depositphotos.com/6168506/24532/i/450/depositphotos_245324140-stock-photo-abstract-blurred-background-landscape-sunset.jpg");
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)  

html_temp = """
    <div style="background-color:transparent;padding:10px;border-radius:5px">
    <h1 style="color:darkblue;text-align:center;font-family:Callibri;"><b>Weed Detection</b></h1>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

file = st.file_uploader("", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import numpy as np
#st.set_option('deprecation.showfileUploaderEncoding', False)
model=tf.keras.models.load_model('code.h5')
model_without_last_2FC = tf.keras.models.Model(model.inputs,model.layers[-5].output)
svm_model_path = "/Volumes/FILES/Weed/Website/svm_classifer.pkl"
with open(svm_model_path,'rb') as svm:
    svm_model = pickle.load(svm)


    
if file is None:
    st.text("")
else:
    image = Image.open(file)
    new_image = image.resize((448, 356))
    st.image(new_image)
    if file:
        temp_dir = tempfile.mkdtemp()
        path = os.path.join(temp_dir, file.name)
        with open(path, "wb") as f:
                f.write(file.getvalue())
    img = cv2.imread(path)
    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb,(224,224))
    feature_of_img = model_without_last_2FC.predict(resized.reshape(1,224,224,3)/255)
    pred = svm_model.predict(feature_of_img.reshape(-1,4096))
    prob = svm_model.predict_proba(feature_of_img.reshape(-1,4096))
    con = np.max(prob) * 100
    conf = con.round(2) 
    col1, col2 = st.columns(2)
    subcol1, subcol2 = col1.columns(2)
    subcol3, subcol4 = col2.columns(2)
    subcol2.success(f"**Prediction:\n {pred[0]}**")
    subcol3.success(f"**Confidence:\n {conf}**")
