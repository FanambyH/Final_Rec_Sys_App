import pandas as pd
import pickle
import numpy as np 

from sklearn.preprocessing import MinMaxScaler

class HrvClassifier():
    """
        This class contains all operations related to stress level analysis
    """
    def __init__(self):
        # hrv stress level classifier
        self.hrv_model_clf = pickle.load(
            open("hrv_classifier_model.sav", 'rb'))

        # model features
        self.features = ['MEAN_RR','SDRR','HR','RMSSD','pNN50','VLF','LF','HF']

        # model level
        self.stress_level = ['low' , 'medium', 'high']

    def scale_data(self,df):
        """
        This function normalizes (scales) the data.

                Parameters:
                        df (DataFrame): the data to scale

                Returns:
                        df (DataFrame): the scaled data
        """
        scaler = MinMaxScaler()
        df[self.features] = scaler.fit_transform(df[self.features])

        return df 

    def predict_proba(self,df):
        """
        This function predicts the stress level probability.

                Parameters:
                        df (DataFrame): the input data (notably the HRV metrics values)

                Returns:
                        label (string): the predicted stress level
                        confidence (float): the probability of the prediction
        """
        pred = self.hrv_model_clf.predict_proba(df[self.features])

        label = self.stress_level[np.argmax(pred)]
        confidence = pred[0][np.argmax(pred)]*100

        return label, confidence

    def predict_stress_level (self,df):
        """
        This function predicts the stress level.

                Parameters:
                        df (DataFrame): the input data (notably the HRV metrics values)

                Returns:
                        pred (int): the predicted stress level
        """
        pred = self.hrv_model_clf.predict(df)[0]
        print("HRV.......")
        print(pred)

        return pred