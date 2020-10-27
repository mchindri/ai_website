import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2

class_names = ['healty', 'k_less']
img = None

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
    return 'Prediction: Healthy plant'
  else:
    return 'Prediction: Plant needs potasium'

def calculate_points(img):
  RECTANGLE_SIZE = 50
  height, width, channels = img.shape
  delta = int(RECTANGLE_SIZE / 2)

  center = (int(width / 2) + ss.x, int(height / 2) + ss.y)
  
  center = (max(0, min(center[0], width)), max(0, min(center[1], height)))
  #st.text(center)

  p0 = [center[0] - delta, center[1] - delta]
  p1 = [center[0] + delta, center[1] + delta]

  if p0[0] < 0:
    p0[0] = 0
    p1[0] = RECTANGLE_SIZE
  if p0[1] < 0:
    p0[1] = 0
    p1[1] = RECTANGLE_SIZE
  if p0[0] > width:
    p0[0] = width
    p1[0] = width - RECTANGLE_SIZE
  if p0[1] > height:
    p0[1] = height
    p1[1] = height - RECTANGLE_SIZE
  return [p0, p1]

def draw_regtangle(img):
  global gp0
  global gp1
  [p0, p1] = calculate_points(img)
  gp0 = p0
  gp1 = p1
  cv2.rectangle(img, (p0[0], p0[1]), (p1[0], p1[1]), (0, 255, 0), 2)
  #st.write((p0[0], p0[1]), (p1[0], p1[1]))
  return img

#		roi = clone[p0[1]:p1[1], p0[0]:p1[0]]
#		miliseconds = int(round(time.time() * 1000))
#		cv2.imwrite('croped_samples\\healty\\%d.png'%miliseconds,roi)
#		cv2.imshow("cropped", roi)

import SessionState
ss = SessionState.get(x=0, y=0)
INCREMENT = 50

def read_buttons():
  if st.sidebar.button('up'):
    ss.y = ss.y - INCREMENT
  if st.sidebar.button('down'):
    ss.y = ss.y + INCREMENT
  if st.sidebar.button('right'):
    ss.x = ss.x + INCREMENT
  if st.sidebar.button('left'):
    ss.x = ss.x - INCREMENT

def local_image():
  img_path = '/content/drive/My Drive/samples/test_samples/img1.jpeg'
  img = keras.preprocessing.image.load_img(img_path)
  img = keras.preprocessing.image.array_to_img(img)
  return img

def upload_image():
  img = st.file_uploader("Choose an image")
  if img is not None:
    img = Image.open(img)
    img = keras.preprocessing.image.array_to_img(img)
    return img

def add_rectangle(img):
    img = keras.preprocessing.image.img_to_array(img)
    img = draw_regtangle(img)
    img = keras.preprocessing.image.array_to_img(img)
    return img

st.title('Up Farm (by Ecoddictive) -- Geek 2020')
read_buttons()
#st.text(get_prediction(img))
img = upload_image()
if img is None:
  img = local_image()
#img = local_image()
if img is not None:
  img = add_rectangle(img)
#img = local_image()
if img is not None:
  st.image(img)

if st.sidebar.button('Get prediction'):
    global gp0
    global gp1
    img = keras.preprocessing.image.img_to_array(img)
    roi = img[gp0[1]:gp1[1], gp0[0]:gp1[0]]
    img_array = tf.expand_dims(roi, 0) # Create a batch
    #st.sidebar.text(roi.shape)
    pred = get_prediction(img_array)
    st.sidebar.text(pred)


