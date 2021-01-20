import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import os
import librosa

import SessionState
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns 


from PIL import Image
from pydub import AudioSegment
from typing import Dict
from hrv_classifier import HrvClassifier
from audio_feature_extractor import AudioFeatureExtractor
from music_recommender import MusicRecommender
from Database import Database



st.set_page_config(layout="wide")
# keep session through pages and layout
session_state = SessionState.get(mood=0, age=0,gender="Male",top_k=[],feedback="low")

##################### HEADER #########################
image = Image.open(f'../image_source/music-makes-me-happy-by-plastickheart-700x500.jpg')
st.sidebar.image(image,use_column_width=True)
# simple description
st.sidebar.title('EMOTION-BASED MUSIC RECOMMENDATION SYSTEM')
st.sidebar.write('This app recommends music based on your stress level and feelings')

session_state.mood = st.sidebar.slider('Indicate your mood level (1: bad - 5: good)',1,5)
session_state.age = st.sidebar.text_input('Enter your age')
session_state.gender =  st.sidebar.selectbox("Select your gender",("Male","Female"))

def encode_gender(gender):
    key = {"Male":0, "Female":1}
    return key[gender]


##################### HEADER #########################


col1,col2 = st.beta_columns(2) 


##################### MUSIC ANALYSIS #########################
def display_col1_description():
    col1.write('MUSIC RECOMMENDATION')
    wave_image = Image.open(f'../image_source/audio_wave.jpg')
    col1.image(wave_image,width=300)
    col1.write()
    col1.write()
    

# File Selection Drop Down
@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}

def file_selector(folder_path):
    filenames = os.listdir(folder_path)
    selected_filename = col1.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

def main():
    fileslist = get_static_store()
    folderPath = col1.text_input('Enter folder path:')

      
    # Declaring the cm variable by the  
    # color palette from seaborn 
    cm = sns.light_palette("blue", as_cmap=True) 

    if folderPath:    
        filename = file_selector(folderPath)
        if not filename in fileslist.values():
            fileslist[filename] = filename
    else:
        fileslist.clear()  # Hack to clear list if the user clears the cache and reloads the page
        col1.info("Select an audio file")

    df_list = pd.DataFrame(columns = ['Title'])

    # clear list
    if col1.button("Clear music list"):
        fileslist.clear()
        df_list = list(fileslist.keys())
    # show  list
    if col1.checkbox("Show music list?", True):
        # transform list into dataframe for ease of use
        df_list = pd.DataFrame(columns = ['Title'])
        df_list['Title'] = list(fileslist.keys())
        # color palette from seaborn 
        cm = sns.light_palette("green", as_cmap=True) 
        col1.dataframe(df_list.style.background_gradient(cmap=cm).set_precision(2))

    # if len(fileslist) > 0:
    #    col1.write(list(fileslist.keys())[0])
    #    play_music(list(fileslist.keys())[0])

    return df_list


def play_music(audio_path):
    audio_file = open(audio_path,'rb')
    audio_bytes = audio_file.read()
    col1.audio(audio_bytes,format='audio/wav')
            
display_col1_description()
fileslist = main()
##################### MUSIC ANALYSIS #########################



##################### MUSIC RECOMMENDATION ###################
# run the recommender and select best top k 
recommender = MusicRecommender()
emotion_clf = AudioFeatureExtractor()
best_pred = pd.DataFrame()
emotion = pd.DataFrame()

if col1.button('Recommend'):
    session_state.top_k = []
    best_pred, emotion = recommender.predict_top_k(list(fileslist['Title']), int(session_state.age), encode_gender(session_state.gender),session_state.mood)
# retrive the  best k track id 
if len(best_pred) != 0 :
    for i in range(len(best_pred)):
        session_state.top_k.append(fileslist['Title'].iloc[best_pred[i]])

    for i in range(len(session_state.top_k)):
        col1.write(session_state.top_k[i].split("\"")[-1])
        fig = px.pie(emotion.T.reset_index(), values=best_pred[i], names='index', color='index',color_discrete_sequence=px.colors.sequential.RdBu,width=400, height=200)
        col1.write(fig)
        play_music(session_state.top_k[i])  

     

##################### MUSIC RECOMMENDATION ###################


##################### HRV STRESS ANALYSIS #########################

col2.write('SHARE YOUR FEELINGS, PROVIDE YOUR FEEDBACK')
hrv_image = Image.open(f'../image_source/HRV.jpeg')
col2.image(hrv_image,width=300)
hrv_file = col2.file_uploader("Upload your HRV metrics for analysis")

if hrv_file is not None:
  hrv_test = pd.read_csv(hrv_file)
  col2.write(hrv_test)

if col2.button('Analyze'):
    # load the model from disk
    hrv_clf = HrvClassifier()
    session_state.feedback,confidence  = hrv_clf.predict_proba(hrv_test)
    
    if session_state.feedback == "low":
        col2.success("Your stress level is " + session_state.feedback + " with "+"%.2f" % confidence  +"% confidence")
    elif session_state.feedback == "medium":
        col2.warning("Your stress level is " + session_state.feedback + " with "+"%.2f" % confidence  +"% confidence")
    else:
        col2.error("Your stress level is " + session_state.feedback + " with "+"%.2f" % confidence  +"% confidence")

##################### HRV STRESS ANALYSIS #########################

##################### SAVE FEEDBACK INTO THE DATABASE #############

# map feedback 
def map_feedback(top_k,feedback):
    df_fb = pd.DataFrame(columns=['title','feedback'])
    df_fb['title'] =  top_k
    df_fb['feedback'] = [feedback] * len(top_k)

    return df_fb

# instatiate the database
db = Database()
         
if col2.button('Send feedback'):
    # map feedback to top k recommendation
    df_feedback = map_feedback(session_state.top_k,session_state.feedback)
    # populate feedback table 
    db.populate_feedback_table(df_feedback)
    col2.write(db.get_feedback_list())

##################### SAVE FEEDBACK INTO THE DATABASE #############