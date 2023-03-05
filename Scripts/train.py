import json
import keras
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.models import load_model
from pathlib import Path
from sklearn.metrics import classification_report


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Function to load the mnist dataset from keras

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Four numpy arrays for the training/test data and labels
    """
    # Load the dataset from keras
    (trainX, trainY), (testX, testY) = mnist.load_data()
    
    # Reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    
    # One hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY
 

def preprocess(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Function to preprocess mnist data

    Args:
        train (np.ndarray): An array containing training data
        test (np.ndarray): An array containing  the test data

    Returns:
        tuple[np.ndarray, np.ndarray]: Two numpy arrays containing the process training and test data
    """
    # Convert from integers to floats
    train_norm = train.astype("float32")
    test_norm = test.astype("float32")
    
    # Normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0

    return train_norm, test_norm


def build_model() -> keras.engine.sequential.Sequential:
    """Function to build a CNN model

    Returns:
        keras.engine.sequential.Sequential: The build and compiled model
    """
    # Initialise a sequential model
    model = Sequential()
    
    # Add 8 layers to the model
    model.add(Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_uniform", input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform"))
    model.add(Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_uniform"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(10, activation="softmax"))
    
    # Create an optimizer
    opt = SGD(learning_rate=0.01, momentum=0.9)

    # Compile model
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

    return model


if __name__ == "__main__":
    # Load the training and test dataset
    trainX, trainY, testX, testY = load_dataset()
    
    # Preprocess the data
    trainX, testX = preprocess(trainX, testX)

    # Build and compile the model
    model = build_model()

    # Fit the model on the training data
    model.fit(trainX, trainY, epochs=3, batch_size=32, verbose=1)
    
    # Save the model to a .h5 file in the app
    model.save(f"{Path(__file__).parents[0]}/../app/data/final_model.h5")

    # Load the model from the app
    model = load_model(f"{Path(__file__).parents[0]}/../app/data/final_model.h5")

    # Get predictions test data   
    predictions = model.predict(testX)
    # Get a single class for each prediction
    predY = np.argmax(predictions, axis=1)
    # Get a single class for each test label
    testYc = np.argmax(testY, axis=1)

    # Evaluate performance of the model
    results = classification_report(testYc, predY, output_dict=True)
    # Create a dictionary containing the results
    score_card = {
        "accuracy": results["accuracy"],
        "precision": results["macro avg"]["precision"],
        "recall": results["macro avg"]["recall"],
        "f1": results["macro avg"]["f1-score"]
    }
    # Save the score card as a json file in the app
    json.dump(score_card, open(f"{Path(__file__).parents[0]}/../app/data/score_card.json", "w"))