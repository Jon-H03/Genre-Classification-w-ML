import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split

import DataPreprocessing.preprocess
from Model import model
from DataPreprocessing import preprocess

# Load the dataset
data = pd.read_csv('../Dataset/SongCSV.csv')

# 2 Preprocess the data (Dropping unnecessary features)
numerical_features = data.select_dtypes(include=['object', 'float64', 'int64']).columns.tolist()
numerical_features.drop(['SongNumber', 'SongID', 'AlbumID', 'AlbumName', 'ArtistID', 'ArtistName', 'Title'], axis=1, inplace=True)
preprocessor = DataPreprocessing.preprocess.DataPreprocessor(numerical_features)

# 3. Split the dataset
X =