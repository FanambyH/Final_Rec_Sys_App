import gym
from gym import spaces
import numpy as np
import pandas as pd

from stable_baselines import A2C
from simulation_environment import SimulationEnvironment
from audio_feature_extractor import AudioFeatureExtractor 

class MusicRecommender():
    """
    This class defines the recommender system pipeline
    """
    
    def __init__(self, base_model='recommender_hrv_based', db='data/simulation_training_hrv_recommender.csv'):
        self.db = pd.read_csv(db)
        self.env = SimulationEnvironment(train=self.db)
        self.model = A2C('MlpPolicy', self.env, verbose=1)
        self.model.load(base_model) 
        self.extractor = AudioFeatureExtractor()
        self.features =[' tenderness', ' calmness', ' power', ' joyful_activation', ' tension', ' sadness', 'classical', 'electronic',
                          'pop', 'rock', ' gender', ' mood',' age_adult', ' age_child', ' age_senior', ' age_young', ' age_young_adult']

    def predict_top_k(self, filenames, user_age=30, user_gender=0,user_mood=1,k=2):
        '''
        This function predicts the best top-k audios.

                Parameters:
                        filenames (string): audio file path
                        user_age (int): user age
                        user_gender (int): user gender (0: Male, 1: Female)
                        user_mood (int): user mood (0:bad-5:good)
                        k (int): the number of top songs to be selected 

                Returns:
                        ranking (list): list of the best top k audios
                        observations (DataFrame): current user observations
        '''
        # Retrieve the observations (state)
        observations = self.pipeline(filenames, user_age, user_gender, user_mood)

        # Predict
        action, _ = self.model.predict(observations, deterministic=True)
        ranking = {"id": [], "action_probability": []}

        # Get the best top k 
        for i, a in enumerate(action):
            ranking['id'].append(i)
            ranking['action_probability'].append(a[0])
        ranking = pd.DataFrame(ranking)
        ranking = ranking.sort_values(by='action_probability', ascending=False)
        ranking = ranking.iloc[:k, :]

        # Get the emotion of the predicted top k
        emotions = [' tenderness', ' calmness', ' power',' joyful_activation', ' tension', ' sadness']
        observations = observations.reset_index()
        observations = observations[emotions]
        observations = observations.iloc[ranking['id'].values.tolist()]

        return ranking['id'].values.tolist(), observations

    def pipeline(self, filenames, user_age=30, user_gender=0, user_mood=1):
        '''
        This function defines the recommendation pipeline.

                Parameters:
                        filenames (list): list of the audio file paths
                        user_age (int): user age
                        user_gender (int): user gender (0: Male, 1: Female)
                        user_mood (int): user mood (0:bad-5:good) 

                Returns:
                        obs (DataFrame): user observation

        '''
        #  Extract Audio Features (genre and emotions)
        df = pd.DataFrame()
        for file in filenames:
            audio = self.extractor.predict(file)
            df = pd.concat((df, audio))
        # Process User Feautres
        def process_age(db):
            if db <= 12:
                return "child"
            elif db > 12 and db <= 25:
                return "young"
            elif db >25 and db <=35:
                return "young_adult"
            elif db > 35 and db <=50:
                return "adult"
            elif db > 50:
                return "senior"
        # Get user Features
        df[' age'] = user_age
        df[' age'] = df[' age'].apply(lambda x: process_age(x), 1)
        df = pd.get_dummies(df, columns=[' age']) 
        # Check if columns doesn't exist and add it 
        ages = [' age_senior', ' age_child',' age_young',' age_young_adult', ' age_adult']
        miss_ages = [a for a in ages if a not in df.columns]
        for m in miss_ages:
            df[m] = 0
        df[' gender'] = user_gender
        df[' mood'] = user_mood
        obs = df[self.features]
        return obs
