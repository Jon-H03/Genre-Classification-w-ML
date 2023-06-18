import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def preprocess(self, data):
        # Further feature selection/engineering if needed
        # You can drop irrelevant features, create new features, etc.
        data = data.drop(['SongNumber',
                          'ArtistID',
                          'SongID',
                          'AlbumID',
                          'ArtistLatitude',
                          'ArtistLongitude',
                          'ArtistLocation',
                          'Danceability',
                          'Year'], axis=1)

        # Drop rows with missing values
        data.dropna(subset=['Genre'], inplace=True)

        # Encoding categorical variables (if applicable)
        # You can use one-hot encoding or label encoding techniques
        # Example with one-hot encoding:
        data = pd.get_dummies(data, columns=['AlbumName', 'ArtistName'])

        # Normalizing numerical features
        numerical_features = ['TimeSignatureConfidence',
                              'TimeSignature',
                              'Duration',
                              'Tempo',
                              'KeySignature',
                              'KeySignatureConfidence']
        data[numerical_features] = self.scaler.fit_transform(data[numerical_features])
        data = data.iloc[:, :9]
        # Save preprocessed data to the same file
        #data.to_csv('preprocessed_data.csv', index=False)
        return data

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    data = pd.read_csv('../SongCSV.csv')
    preprocessor.preprocess(data)



