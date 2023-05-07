import streamlit as st
from predict_page_knn import show_predict_page_knn
import pandas as pd
import re

side = st.sidebar.selectbox("Select the model",("KNN",))

if side == "KNN":
    show_predict_page_knn()