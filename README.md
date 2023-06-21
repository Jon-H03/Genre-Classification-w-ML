# Genre classification with ML.
## A project built with PyTorch
 
### Introduction
This project aims to classify genres, given numerical information like key and time signatures, duration, and danceability. This machine learning model was done simply out of curiosity and wanting to gain some experience implementing my own model with PyTorch. 

### Dataset
The dataset file contains 3 files: `extract.py`, `hdf5_getters.py` (written by the creators of the dataset), and data from the 10,000 song dataset, a subset of the 1,000,000 song dataset. It was discovered through extracting the data from the tracks in the dataset that there is no genre data within the dataset, a quality I aimed to predict, so I had to seek external methods and get the genre data through last.FM's API. The data is then written to a file `SongCSV.csv`, where the preprocessor will organize it and prepare it to be trained.

### DataPreprocessing
The data preprocessing was difficult because I had many songs without information on genre, or other numerical qualities that may have been missing. To make for an attempt at relatively unbiased training, I simply dropped any rows that had missing information or probably were not relevant to the model, however, obviously, this comes with some complications of its own. The preprocessor will write to a file `preproccessed_data.csv` but will also simply return all the data in a pandas DataFrame object.

### Model
`model.py` contains the model which consists of 5 layers, 4 linear, and a rectified linear unit. Initially, it had only 2 linear layers with a ReLU one, however, the performance was not as great. Using more layers gives the model increased capacity, hierarchical feature extraction, non-linear transformations, and improved representations of data. However, adding more layers may not always be the answer as it introduces challenges such as the need for more computational power and more data.

### Training

