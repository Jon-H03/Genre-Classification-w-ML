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
The training process is implemented in the `train.py` script. This script brings together the preprocessed data, the model architecture, and the necessary training routines. During training, the model learns to classify music genres by iteratively adjusting its weights and biases based on the provided labeled data. The training script utilizes techniques such as mini-batch gradient descent and backpropagation to optimize the model's parameters. Additionally, it includes evaluation metrics to monitor the model's performance and to prevent overfitting, such as validation loss and accuracy. Once training is complete, the trained model parameters can be saved for future use or evaluation.

### Conclusion
In conclusion, this project demonstrates the application of machine learning techniques for music genre classification. By leveraging a dataset of numerical song attributes and external genre information, we developed a PyTorch model capable of predicting music genres based on these features. The process involved data preprocessing to handle missing or irrelevant information, model architecture design with multiple layers for improved representation learning, and training the model using gradient-based optimization algorithms. Through this project, valuable insights were gained into the challenges and considerations involved in multi-class music genre classification using machine learning.

### Future Work
While this project lays the foundation for music genre classification, there are several avenues for future exploration. Here are a few potential areas of improvement I'd consider:
- Incorporating audio features: Currently, the model relies solely on numerical attributes. Integrating audio features, such as spectrograms or MFCCs, could provide additional information and potentially improve classification performance.
- Ensemble methods: Exploring ensemble learning techniques, such as combining multiple models or implementing bagging/boosting algorithms, could enhance the overall accuracy and robustness of genre classification.
- Fine-tuning hyperparameters: Further optimization of hyperparameters, such as learning rate, regularization strength, or batch size, may yield better model performance. Techniques like grid search or Bayesian optimization can be employed for this purpose.
