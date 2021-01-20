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
st.header('MUSIC RECOMMENDATION')
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


##################### MUSIC ANALYSIS #########################
col1,col2 = st.beta_columns(2)  
col2.header("RECOMMENDED FOR YOU")

def display_col1_description():
    """This functions displays the descripion of the recommender system
                    Parameters:
                            -
                            
                    Returns:
                            -
    """
    wave_image = Image.open(f'../image_source/audio_wave.jpg')
    col1.image(wave_image,width=300)
    col1.write()
    col1.write()
    

# File Selection Drop Down
@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded
                    Parameters:
                            -
                            
                    Returns:
                            -
    """
    return {}

def file_selector(folder_path):
    """
        This function allows to select file using a fileselector 

                Parameters:
                        folder_path (string): the folder root path
                        
                Returns:
                        path (string): the selected file path
                        
    """
    filenames = os.listdir(folder_path)
    selected_filename = col1.selectbox('Select a file', filenames)
    path = os.path.join(folder_path, selected_filename)
    return path

def main():
    """
        This function displays all the gui elements of the music recommender system

                Parameters:
                        -

                Returns:
                        df_list (DataFrame): the list of input audio entered by the user
                        
    """
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


    return df_list


def play_music(audio_path):
    """
        This function displays and plays music on the screen 

                Parameters:
                        auio_path (string): the audio file path

                Returns:
                        -
                        
    """
    audio_file = open(audio_path,'rb')
    audio_bytes = audio_file.read()
    col2.audio(audio_bytes,format='audio/wav')

# display the music recommender gui         
display_col1_description()
fileslist = main()
##################### MUSIC ANALYSIS #########################



##################### MUSIC RECOMMENDATION ###################
# run the recommender and select best top k 
recommender = MusicRecommender()
emotion_clf = AudioFeatureExtractor()
best_pred = pd.DataFrame()
emotion = pd.DataFrame()

# recommend the music 
if col1.button('Recommend'):
    session_state.top_k = []
    best_pred, emotion = recommender.predict_top_k(list(fileslist['Title']), int(session_state.age), encode_gender(session_state.gender),session_state.mood)

# retrive the  best k track id 
if len(best_pred) != 0 :
    for i in range(len(best_pred)):
        session_state.top_k.append(fileslist['Title'].iloc[best_pred[i]])

    for i in range(len(session_state.top_k)):
        col2.markdown('<span class="badge badge-pill badge-success">'+session_state.top_k[i].split('\\')[-1]+ '</span>',unsafe_allow_html=True)
        fig = px.pie(emotion.T.reset_index(), values=best_pred[i], names='index', color='index',color_discrete_sequence=px.colors.sequential.RdBu,width=400, height=200)
        col2.write(fig)
        play_music(session_state.top_k[i])  

     

##################### MUSIC RECOMMENDATION ###################


##################### HRV STRESS ANALYSIS #########################

st.header('SHARE YOUR FEELINGS, PROVIDE YOUR FEEDBACK')
hrv_image = Image.open(f'../image_source/HRV.jpeg')
st.image(hrv_image,width=300)
hrv_file = st.file_uploader("Upload your HRV metrics for analysis")

# load the model from disk
hrv_clf = HrvClassifier()

if hrv_file is not None:
  hrv_test = pd.read_csv(hrv_file)
  st.write(hrv_test[hrv_clf.features])

# Analyze stress level
if st.button('Analyze'):
    session_state.feedback,confidence  = hrv_clf.predict_proba(hrv_test)
    
    if session_state.feedback == "low":
        st.success("Your stress level is " + session_state.feedback + " with "+"%.2f" % confidence  +"% confidence")
    elif session_state.feedback == "medium":
        st.warning("Your stress level is " + session_state.feedback + " with "+"%.2f" % confidence  +"% confidence")
    else:
        st.error("Your stress level is " + session_state.feedback + " with "+"%.2f" % confidence  +"% confidence")

##################### HRV STRESS ANALYSIS #########################

##################### SAVE FEEDBACK INTO THE DATABASE #############
  
def map_feedback(top_k,feedback):
    """
        This function maps the given feedback to the proposed music 

                Parameters:
                        top_k (list): list of the best top_k music
                        feedback (string): the user feedback (stress level)

                Returns:
                        df_fb (DataFrame): the feedback data ready to be saved
                        
    """
    df_fb = pd.DataFrame(columns=['title','feedback'])
    df_fb['title'] =  top_k
    df_fb['feedback'] = [feedback] * len(top_k)

    return df_fb

# instatiate the database
db = Database()
         
if st.button('Send feedback'):
    # map feedback to top k recommendation
    df_feedback = map_feedback(session_state.top_k,session_state.feedback)
    # save feedback to the database 
    db.populate_feedback_table(df_feedback)
    st.write(db.get_feedback_list())

##################### SAVE FEEDBACK INTO THE DATABASE #############