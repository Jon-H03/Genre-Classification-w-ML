import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    def __init__(self, numerical_features):
        self.numerical_features = numerical_features
        self.scaler = MinMaxScaler()

    def preprocess(self, data):
        # Drop rows with missing values
        data.dropna(inplace=True)

        # Encoding categorical variables (if applicable)
        # You can use one-hot encoding or label encoding techniques
        # Example with one-hot encoding:
        data = pd.get_dummies(data, columns=['AlbumName', 'ArtistName'])

        # Normalizing numerical features
        # Example with Min-Max scaling:
        data[self.numerical_features] = self.scaler.fit_transform(data[self.numerical_features])

        # Further feature selection/engineering if needed
        # You can drop irrelevant features, create new features, etc.
        data = data.drop(['SongNumber', 'SongID', 'AlbumID'], axis=1)

        return data


