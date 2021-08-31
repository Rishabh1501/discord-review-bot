"""
Copyright (c) 2021 Rishabh Kalra <rishabhkalra1501@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# importing libraries
import os
import pickle
import joblib
import numpy as np
import pandas as pd

# importing custom packages
from data_cleaning.data_cleaning import Cleaner


class PredictAPI:
    """
    Class to Predict and classify the sentences
    
    Keyword arguments: 
        prediction_folder_path="Prediction_Data"
        training_folder_path="Training_Data",
        log_folder_name="Prediction_Logs",
        log_file_name="3-prediction.txt"
    
    argument -- 
        prediction_folder_path: Path to store the csv predictions (only valid if input given is a csv file).
        training_folder_path: Folder path where models are stored.
        log_folder_name: Specifies the folder for Training Logs.
        log_file_name: Specifies the name of the log file.
    
    Return: None
    """
    
    def __init__(self, prediction_folder_path="Prediction_Data", training_folder_path="Training_Data"):
        self.prediction_folder_path = prediction_folder_path
        self.training_folder_path = training_folder_path
        self.cleaner = Cleaner()

        if prediction_folder_path not in os.listdir():
            os.mkdir(prediction_folder_path)


    def clean_sentence(self, sentence):
        """
        cleans the sentence for prediction

        Args:
            sentence (string): sentence to be cleaned

        Raises:
            Exception: any Exception, check logs for specifics

        Returns:
            string: cleaned sentence
        """
        try:
            sentence = self.cleaner.review_to_words(sentence)
            return sentence
        except Exception as e:
            raise Exception(e)

    def predict_model_sentence(self, sentence, model_name="svc_model.sav", vector_model="vectorize.pickle"):
        """
        predicts the output and returns it

        Args:
            sentence (string): sentence to be predicted.
            model_name (str, optional): name of the prediction model inside Training_Data Folder. Defaults to "model.sav".
            vector_model (str, optional): name of the vector model inside Training_Data Folder. Defaults to "vectorize.pickle".

        Raises:
            Exception: any Exception, check logs for specifics
            
        Returns:
            str: predicted data
        """
        try:
            df = pd.DataFrame([sentence], columns=['review'])

            vector_path = os.path.join(self.training_folder_path, vector_model)
            model_path = os.path.join(self.training_folder_path, model_name)
            with open(vector_path, 'rb') as f:
                vector = pickle.load(f)

            nb_model = joblib.load(model_path)

            x_vector = vector.transform(df['review'])
            # print(x_vector)
            y_predict = nb_model.predict(x_vector)
            print(y_predict)
            # predicted_probability = nb_model.predict_proba(x_vector)
            # if list(predicted_probability.flatten())[0] == list(predicted_probability.flatten())[1]:
            #     print("UNKNOWN")
            # elif list(predicted_probability.flatten())[np.argmax(predicted_probability)] > 0.6:
            # else:
            #     print("UNKNOWN")

            return y_predict[0]

        except Exception as e:
            raise Exception(e)
