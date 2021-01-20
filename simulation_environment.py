import gym
from gym import spaces
import numpy as np
import pandas as pd

  

from stable_baselines3 import A2C
from hrv_classifier import HrvClassifier

class SimulationEnvironment(gym.Env):
    """
        This class is the music recommender system simulation environment
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, train, hrv_interpreter='hrv_classifier_model.sav'):
        super( SimulationEnvironment, self).__init__()

        # model features
        self.features  = [' tenderness', ' calmness', ' power', ' joyful_activation', ' tension', ' sadness', 'classical', 'electronic',
                          'pop', 'rock', ' gender', ' mood',' age_adult', ' age_child', ' age_senior', ' age_young', ' age_young_adult']
        # shuffle the date
        self.data =  train.copy().sample(frac=1, random_state=47)
        # set Actions space : Propose audio or not
        self.action_space = spaces.Box(
            low=0, high=1
            , shape=(1,), dtype=np.float16)
        # Set observation space
        self.observation_space = spaces.Box(
            low=0, high=1
            , shape=(len(self.features),), dtype=np.float16)
        
        # hrv classifier model
        self.hrv_interpreter = HrvClassifier().hrv_model_clf
        # Initial step 
        self.current_step = 0
        self.sum_reward = 0 
        
        # hrv features 
        self.hrv_features = ['MEAN_RR','SDRR','HR','RMSSD','pNN50','VLF','LF','HF']

        # label
        self.label = 'ratings'


    def get_next_observation(self):
        '''
        This functions gets the next observation from the simulation dataset

                Parameters:
                        -

                Returns:
                        obs (DataFrame): the next observation
        '''
        obs = self.data[self.features].iloc[self.current_step].values
        return obs

    def reset(self):
        '''
        This function resets the steps

                Parameters:
                        -

                Returns:
                        obs (DataFrame): the next observation
        '''
        self.data =  self.data.copy().sample(frac=1, random_state=47)
        self.current_step = 0
        self.sum_reward = 0 
        self.accuracy_score = 0 
        obs = self.get_next_observation()
        return obs

    def step(self, action):
        '''
        This function defines one learning step.

                Parameters:
                        action (float): between 0 and 1 : probability to recommend the audio 

                Returns:
                        obs (DataFrame): the next observation
                        done (Boolean):  defines if the learning step is complete or not
                        reward (int): the reward 
        '''
        # Retrieve user hrv 
        user_hrv = self.data[self.hrv_features].iloc[self.current_step].to_frame().T    
        # Interpret HRV
        user_stress = self.hrv_interpreter.predict_stress_level(user_hrv)
        # Add Some reward (+1) if the model propose  song that user likes 
        if user_stress == 0 and action[0] > 0.5: 
            reward = 1  * action[0]  
        elif user_stress !=0 and action[0] < 0.5: 
            reward = 1  * (1 - action[0])  
        else:
            reward = 0  #- 1 *  action[0] 
        
        
        if self.current_step > self.data.shape[0] - 10:
            done = True 
        else:
            self.current_step += 1 
            done = False 
        obs = self.get_next_observation()
        self.sum_reward += reward 
        self.mean_reward = self.sum_reward /  self.current_step 
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        '''
        This function renders the environment to the screen.

                Parameters:
                        -

                Returns:
                        -
        '''
        # Render the environment to the screen
        if self.current_step % 10 == 0:
            print(f'Sum Observation:{self.current_step}, Sum reward:{self.sum_reward}, Success Rate:{self.sum_reward / self.current_step}')
           