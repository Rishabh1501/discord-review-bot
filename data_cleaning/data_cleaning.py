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

import os
import nltk
import string
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
# nltk.download()  To download all the nltk libraries


class Cleaner:
    """
    Class to clean the data , perform stemming and preparing the data for cleaning

    Keyword arguments: log_folder_name="Training_Logs", log_file_name="2-data_cleaner.txt"

    argument -- 
        log_folder_name: Specifies the folder for Training Logs
        log_file_name: Specifies the name of the log file

    Return: None
    """

    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stop_words = stopwords.words('english')
        self.unnecessary_words = ["br", "'ll",
                                  "..", "....", "n't", "...", " ... "]
        self.punctuation = string.punctuation

    def review_to_words(self, sentence):
        """
        Converts a sentence into a clean and stemmed sentence

        Args:
            sentence (string): sentence to be cleaned

        Raises:
            Exception: any Exception, check logs for specifics

        Returns:        
            String : Cleaned Sentence
        """
        try:
            words = nltk.word_tokenize(sentence)
            words_list = list()
            for word in words:
                word = word.lower()
                letter_list = list()
                # print(word)
                if word not in self.stop_words:
                    if word not in self.unnecessary_words:
                        for letter in word:
                            if letter not in self.punctuation:
                                letter_list.append(letter)
                        if letter_list:
                            word = ''.join(letter_list)
                            words_list.append(self.stemmer.stem(word))
            return " ".join(words_list)

        except Exception as e:
            raise Exception(e)

    def ret_cleaned_dataframe(self, dataframe, col_num=0):
        """Returns a cleaned dataframe

        Args:
            dataframe (pandas.DataFrame): DataFrame to be Cleaned
            col_num (int, optional): Number of the column to be cleaned. Defaults to 0.

        Raises:
            Exception: any Exception, check logs for specifics

        Returns:
            pandas.DataFrame: pandas DataFrame
        """
        try:
            col = dataframe.columns

            # dataframe[col[col_num+1]] = dataframe[col[col_num+1]].apply(lambda x: 1 if x == "positive" else 0)
            return dataframe

        except Exception as e:
            raise Exception(e)

    def save_dataframe_in_csv(self, dataframe, file_path):
        """saves the dataframe in csv format

        Args:
            dataframe (pandas.DataFrame): DataFrame to be saved
            file_path (string/path): path to save the dataframe in csv format

        Raises:
            Exception: any Exception, check logs for specifics
        """
        try:
            dataframe.to_csv(file_path, index_label=False)
        except Exception as e:
            raise Exception(e)
