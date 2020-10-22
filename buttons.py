import streamlit as st
import SessionState

ss = SessionState.get(x=0, y=0)

INCREMENT = 10

def read_buttons():
  if st.sidebar.button('up'):
    ss.x = ss.x - INCREMENT
  if st.sidebar.button('down'):
    ss.x = ss.x + INCREMENT
  if st.sidebar.button('right'):
    ss.y = ss.y + INCREMENT
  if st.sidebar.button('left'):
    ss.y = ss.y - INCREMENT
  st.text('x=%d - y=%d'%(ss.x, ss.y))

read_buttons()