import librosa
import pandas as pd
import pickle
import numpy as np 

from pydub import AudioSegment

class AudioFeatureExtractor():
    """ This class contains all operations related to audio files
    """
    def __init__(self):
        # Model in charge of Detecting Music Genre
        self.model = pickle.load(
            open('model_saved/emo_genre_clf_model.sav', 'rb'))    

        # Utility Scaler
        self.scaler = pickle.load(open('model_saved/Scaler_Extractor.sav', 'rb'))
        
        # List of Audio Features
        self.features = ['chroma_sftf', 'rolloff', 'zero_crossing_rate', 'rmse', 'flux', 'contrast', 'flatness',
                          'sample_silence', 'mfcc_0', 'mfcc_1','mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5', 'mfcc_6',
                          'mfcc_7', 'mfcc_8', 'mfcc_9', 'mfcc_10','mfcc_11','mfcc_12', 'mfcc_13', 'mfcc_14',
                          'mfcc_15', 'mfcc_16', 'mfcc_17', 'mfcc_18', 'mfcc_19', 'tempo']
                          
        #List of labels
        self.labels =  [' tenderness', ' calmness', ' power',' joyful_activation', ' tension', ' sadness',
           "classical", "electronic", "pop", "rock"]

    def read_audio(self, audiofile, debug=True):
        """
        This function reads an audio file.

                Parameters:
                        audiofile (string): the audio file path

                Returns:
                        audio_features (DataFrame): the extracted audio features
        """
        # Empty array of dicts with the processed features from all files
        arr_features = []
        # Read the audio file
        audiofile = self.convert_audio_to_wav(audiofile)
        signal, sr = librosa.load(audiofile, duration=30)
        # pre-emphasis before extracting features
        signal_filt = librosa.effects.preemphasis(signal)
        track_id = audiofile.replace(".wav", "")
        # Append the result to the data structure
        features = self.extract_audio_features(signal_filt, sr, track_id)
        arr_features.append(features)
        # Return DataFrame of audio Features
        audio_features = pd.DataFrame(arr_features).fillna(0)
        
        return audio_features

    def extract_audio_features(self, y, sr, id):
        '''
        This function extracts audio features from an audio file.

                Parameters:
                        id (string): the audio track id 
                        y 
                        sr 

                Returns:
                        audio_features (DataFrame): the extracted audio features
        '''
        # Features to concatenate in the final dictionary
        features = {'chroma_sftf': None, 'rolloff': None, 'zero_crossing_rate': None, 'rmse': None,
                    'flux': None, 'contrast': None, 'flatness': None}

        # Count silence
        if 0 < len(y):
            y_sound, _ = librosa.effects.trim(y)
        features['sample_silence'] = len(y) - len(y_sound)            

        # Using librosa to calculate the features
        features['chroma_sftf'] = np.mean(
            librosa.feature.chroma_stft(y=y, sr=sr))
        features['rolloff'] = np.mean(
            librosa.feature.spectral_rolloff(y, sr=sr))
        features['zero_crossing_rate'] = np.mean(
            librosa.feature.zero_crossing_rate(y))
        features['rmse'] = np.mean(librosa.feature.rms(y))
        features['flux'] = np.mean(librosa.onset.onset_strength(y=y, sr=sr))
        features['contrast'] = np.mean(
            librosa.feature.spectral_contrast(y, sr=sr))
        features['flatness'] = np.mean(librosa.feature.spectral_flatness(y))

        # MFCC treatment
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for idx, v_mfcc in enumerate(mfcc):
            features['mfcc_{}'.format(idx)] = np.mean(v_mfcc)

        features['tempo'] = librosa.beat.tempo(y, sr=sr)[0]
        features['track_id'] = id
        return features

    
    def predict(self, audiofile,return_df=True):
        '''
        This function predicts the music genre

                Parameters:
                        audiofile (string): the audio file path

                Returns:
                        genre (DataFrame): probability of music genre
        '''
       # Extract audio features
        audio_features = self.read_audio(audiofile)
        audio_features = audio_features[self.features]
        audio_features = self.scaler.transform(audio_features)
        # Predict genre and emotion
        prediction = self.model.predict_proba(audio_features)
        if return_df:
            prediction = pd.DataFrame(prediction)
            prediction.columns = self.labels
        return prediction
            

    def convert_audio_to_wav(self,src):
        '''
        This function converts any mp3 file into wav format
                Parameters:
                        src (string): audio file source (path)

                Returns:
                        dst (string): new source of the converted audio file
        '''
        # Convert mp3 to wav
        if "mp3" in src:
            sound = AudioSegment.from_mp3(src)
            dst = src.replace('mp3','wav')
            sound.export(dst, format="wav")
        else: 
            dst = src
        
        return dst
