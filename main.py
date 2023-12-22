import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import cv2
import os

from skimage import io
from tensorflow.keras.models import load_model

@st.cache_data
def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
    
    faces = face_classifier.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=4,
    minSize=(20, 20)
    )
    
    return faces

@st.cache_data
def draw(img, face):
    (x,y,w,h) = face
    mask_label = {0:'NO MASK!',1:'Mask'}
    label_color = {0: (255,0,0), 1: (0,255,0)}
    
    crop = img[y:y+h,x:x+w]
    
    crop = cv2.resize(crop,(128, 128))
    crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = np.reshape(crop,[1,128,128,1]) / 255.0
    
    mask_result = model.predict(crop)
            
    pred_label = round(mask_result[0][0])
            
    cv2.putText(img,mask_label[pred_label],
                (x, y+90), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, label_color[pred_label], 2)
            
    cv2.rectangle(img,(x,y),(x+w,y+h), 
                label_color[pred_label],2)
    
    return img

@st.cache_data
def detect_mask(file): 
    img = io.imread(file)
    faces = detect_faces(img)
    
    if len(faces)>=1:
        for i in range(len(faces)):
            draw(img, faces[i])
                        
        return img
            
    else:
        return None

@st.cache_resource
def models_load():
    return load_model('models/best_model.keras'), cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')


model, face_classifier = models_load()

st.title("Масочный режим")

st.write("""
        Вы загружаете фото, мы показываем плохих людей, которые маски не носят.
        
        Если мы не нашли всех людей на фото
        или определили что-то неправильно, то не обижайтесь. Мы не всесильны, а вы придирчивы 🐹
        """)

st.write("\n\n")
st.write("\n\n ##### Загрузите фото")

file = st.file_uploader(label="Я загружаю фотографии", type=['png', 'jpg'])
if file:
    st.image(file, caption="Вот это вы загрузили")

bopton = st.button(label="Найти их", disabled=False, key=1)

if bopton:
    result = detect_mask(file)

    if result is None:
        st.write("А тут никого нет 👀")

    else:
        st.image(result, caption="Вот это вы получаете")      
