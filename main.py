import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.

#streamlit run main.py
#streamlit run https://raw.githubusercontent.com/streamlit/demo-uber-nyc-pickups/master/app.py

st.title('Up Farm (by Eccodictive) -- Geek 2020')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class_names = ['healty', 'k_less']
import matplotlib.pyplot as plt
import numpy as np

class_names = ['healty', 'k_less']

img_path = '/content/drive/My Drive/samples/test_samples/img1.jpeg'
img = keras.preprocessing.image.load_img(img_path)

img_array = keras.preprocessing.image.img_to_array(img)
img_array = img_array[0:50, 0:50,:]

ni = np.array(img_array)

st.image(ni.astype(int))

model = tf.keras.models.load_model('/content/drive/My Drive/Geek/ai_model')
img_array = tf.expand_dims(img_array, 0) # Create a batch

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(img_array)
idx = np.argmax(predictions)
if idx == 0:
  st.text('Healthy plant')
else:
  st.text('Plant needs potasium')