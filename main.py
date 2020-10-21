import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

class_names = ['healty', 'k_less']

def get_image():
  img_path = '/content/drive/My Drive/samples/test_samples/img1.jpeg'
  img = keras.preprocessing.image.load_img(img_path)
#  st.image(img)
  img_array = keras.preprocessing.image.img_to_array(img)
  img_array = img_array[0:50, 0:50,:]

  ni = np.array(img_array)
  #print image
  #st.image(ni.astype(int))
  img_array = tf.expand_dims(img_array, 0) # Create a batch
  return img_array

def get_prediction(img_array):
  model = tf.keras.models.load_model('/content/drive/My Drive/Geek/ai_model')

  probability_model = tf.keras.Sequential([model, 
                                          tf.keras.layers.Softmax()])
  predictions = probability_model.predict(img_array)
  idx = np.argmax(predictions)
  if idx == 0:
    return 'Healthy plant'
  else:
    return 'Plant needs potasium'

def upload_image():
  img = st.file_uploader("Choose an image")
  if img is not None:
    img = Image.open(img)
    img = keras.preprocessing.image.array_to_img(img)
    st.image(img)

st.title('Up Farm (by Eccodictive) -- Geek 2020')
#img = get_image()
#st.text(get_prediction(img))
upload_image()


