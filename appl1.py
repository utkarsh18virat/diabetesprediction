# Import the required libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import os

# Define the relative path for the dataset and model
data_path = 'diabetes.csv'
model_path = 'trained_model.sav'

# Load the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv(data_path)

# Print the first 5 rows of the dataset
print(diabetes_dataset.head())

# To get the number of rows and columns in the dataset
print(diabetes_dataset.shape)

# To get the statistical measures of the data
print(diabetes_dataset.describe())

# To get details of the outcome column
print(diabetes_dataset['Outcome'].value_counts())

# Separating the data and labels
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

# To print the independent variables
print(X)

# To print the outcome variable
print(Y)

# Split the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

# Build the model
classifier = svm.SVC(kernel='linear')

# Train the support vector Machine Classifier
classifier.fit(X_train, Y_train)

# Accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data : ', training_data_accuracy)

# Accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data : ', test_data_accuracy)

input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)

# Change the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')

# Save the trained model
with open(model_path, 'wb') as file:
    pickle.dump(classifier, file)

# Load the saved model
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)
