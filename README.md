# Genre classification with ML.
## A project built with PyTorch
 
### Introduction
This project is an exploration into genre classification using machine learning. Leveraging PyTorch, it classifies songs based on features such as key signatures, duration, and danceability. This endeavor was driven by curiosity and the desire to gain hands-on experience with PyTorch.

### Dataset
The primary data source is a 10,000 song subset of the Million Song Dataset. Due to the absence of genre information in this dataset, genre data was supplemented using last.FM's API. The extract.py and hdf5_getters.py scripts facilitate data extraction and structuring into SongCSV.csv for preprocessing.

### DataPreprocessing
The preprocessing stage was challenging due to incomplete data entries. Rows with missing genre or other vital information were omitted to maintain data integrity. The preprocessed data is stored in preprocessed_data.csv and also available as a pandas DataFrame for model training.

### Model
The model, defined in model.py, consists of five layers: four linear layers and one rectified linear unit. The multi-layered structure enhances the model's ability to capture complex patterns and relationships in the data.

### Training
Training is executed in train.py, combining preprocessed data with the model architecture. Techniques like mini-batch gradient descent and backpropagation optimize the model. Evaluation metrics ensure accuracy and guard against overfitting.

### Conclusion
This project showcases the use of machine learning for music genre classification. It highlights the intricacies of working with multi-class datasets and the importance of a robust preprocessing and training strategy.

### Future Work
While this project lays the foundation for music genre classification, there are several avenues for future exploration. Here are a few potential areas of improvement I'd consider:
- Incorporating audio features: Currently, the model relies solely on numerical attributes. Integrating audio features, such as spectrograms or MFCCs, could provide additional information and potentially improve classification performance.
- Ensemble methods: Exploring ensemble learning techniques, such as combining multiple models or implementing bagging/boosting algorithms, could enhance the overall accuracy and robustness of genre classification.
- Fine-tuning hyperparameters: Further optimization of hyperparameters, such as learning rate, regularization strength, or batch size, may yield better model performance. Techniques like grid search or Bayesian optimization can be employed for this purpose.
