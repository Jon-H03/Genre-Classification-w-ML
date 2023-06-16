import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Load the data from CSV into a DataFrame
data = pd.read_csv('../Dataset/SongCSV.csv')

# Drop rows with missing values
data.dropna(inplace=True)

# Encoding categorical variables (if applicable)
# You can use one-hot encoding or label encoding techniques
# Example with one-hot encoding:
data = pd.get_dummies(data, columns=['AlbumName', 'ArtistName'])

# Normalizing numerical features
# Example with Min-Max scaling:
scaler = MinMaxScaler()
numerical_features = ['Duration', 'KeySignature', 'KeySignatureConfidence', 'Tempo', 'TimeSignature', 'TimeSignatureConfidence']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# Splitting the data into training and validation sets
# Adjust the test_size and random_state parameters as needed
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

# Further feature selection/engineering if needed
# You can drop irrelevant features, create new features, etc.

train_data = train_data.drop(['SongNumber', 'SongID', 'AlbumID'], axis=1)
val_data = val_data.drop(['SongNumber', 'SongID', 'AlbumID'], axis=1)

