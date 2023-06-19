import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from Model.model import RecommendationModel
from DataPreprocessing.preprocess import DataPreprocessor

# Load the dataset
data = pd.read_csv('../SongCSV.csv')

# Initialize the DataPreprocessor
preprocessor = DataPreprocessor()

# Preprocess the data
preprocessed_data = preprocessor.preprocess(data)

# Drop rows with missing values
preprocessed_data = preprocessed_data.dropna(subset=['Genre'])

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(preprocessed_data, test_size=0.2, random_state=42)

# Initialize label encoder
label_encoder = LabelEncoder()
label_encoder.fit(train_data['Genre'])

# Convert the data into tensors
train_inputs = torch.tensor(train_data.drop(['Genre', 'Title'], axis=1).values, dtype=torch.float32)
train_targets = torch.tensor(label_encoder.fit_transform(train_data['Genre']), dtype=torch.long)
test_inputs = torch.tensor(test_data.drop(['Genre', 'Title'], axis=1).values, dtype=torch.float32)
test_targets = torch.tensor(label_encoder.fit_transform(test_data['Genre']), dtype=torch.long)

# Decode the encoded target values
train_genre_labels = label_encoder.inverse_transform(train_targets)

# Print the unique target values
unique_genres = set(train_genre_labels)
print("Unique genres:", len(unique_genres))
print("Encoded target values:", train_targets)

# Create our model with respective genres in list.
genres = ['rock', 'pop', 'metal', 'jazz', 'classical', 'hip-hop', 'edm', 'dance', 'rap',
          'r&b', 'reggae', 'alternative', 'indie', 'punk', 'folk', 'country', 'electronic']

# Get our # of features for our model
initial_features = train_inputs.shape[1]
print(initial_features)
# Set up our model and device-agnostic code
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RecommendationModel(initial_features).to(device)

# Set random seed (for now)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Define Hyperparameters
learning_rate = 0.001
batch_size = 32
num_epochs = 100

# Define the loss function & optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create the train_dataset and test_dataset
train_dataset = TensorDataset(train_inputs, train_targets)
test_dataset = TensorDataset(test_inputs, test_targets)

# Create the DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
for epoch in range(num_epochs):
    model.train()

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(inputs)
        print("Outputs shape:", outputs.shape)
        print("Labels shape:", labels.shape)

        # Calculate loss
        loss = loss_fn(outputs, labels)

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backwards & optimizer step
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        total_test_loss = 0.0
        total_test_samples = 0
        correct_predictions = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate test loss
            test_loss = loss_fn(outputs, labels)
            total_test_loss += test_loss.item()

            # Count correct predictions
            _, predicted_labels = torch.max(outputs, 1)
            correct_predictions += (predicted_labels == labels).sum().item()

            total_test_samples += labels.size(0)

        average_test_loss = total_test_loss / len(test_loader)
        accuracy = correct_predictions / total_test_samples

    # Print progress
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss.item():.5f} | Test Loss: {test_loss.item():.5f}")
