import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import altair as alt
import time
import zipfile

# Page title
st.set_page_config(page_title='NMI Breast Cancer Classifier', page_icon='🤖')
st.title('🤖 NMI Breast Cancer Classifier')

with st.expander('About this Demo'):
  st.markdown('**What can this Demo do?**')
  st.info('This demo allows users to upload breast ultrasound images and utilizes a deep learning model for breast cancer detection.')

  st.markdown('**How to use the demo?**')
  st.warning('Please upload an image of a breast ultrasound to get started. The model will then predict the likelihood of breast cancer based on the uploaded image.')

#   st.markdown('**Under the hood**')
#   st.markdown('Data sets:')
#   st.code('''- Drug solubility data set
#   ''', language='markdown')
  
#   st.markdown('Libraries used:')
#   st.code('''- Pandas for data wrangling
# - Scikit-learn for building a machine learning model
# - Altair for chart creation
# - Streamlit for user interface
#   ''', language='markdown')


# Sidebar for accepting input parameters
with st.sidebar:
    # Load data
    st.header('1.1. Input data')

    st.markdown('**1. Use custom data**')
    uploaded_file = st.file_uploader("Upload an image of breast ultrasound", type=['png','jpg','webp','jpeg'])

    submit = st.button('Submit and Run Model')
      

def inference(image):
    # Load the model
    model = None

    # Preprocess the image
    image = None

    # Make predictions
    prediction = None

    return prediction

prediction_dict = {
    'Benign': 0.2,
    'Malignant': 0.8    
}

# Initiate the model building process
import numpy as np
if uploaded_file and submit: 
    # show image
    st.markdown('**Uploaded Image**')
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # inference
    # prediction_dict = inference(uploaded_file)
    st.markdown('**Model Prediction**')
    for key, value in prediction_dict.items():
        if value == max(prediction_dict.values()):
            st.markdown(f'**{key}**: {value}')
        else:
            st.markdown(f'{key}: {value}')
else:
    st.warning('👈 Upload a breast ultrasound image or click *"Load example data"* to get started!')
