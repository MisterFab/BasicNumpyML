import pandas as pd
import numpy as np
import datetime
from models.classification import (Perceptron, AdalineGD, AdalineSGD, 
                                  LogisticRegressionGD, KNearestNeighbors)
from models.tree import DecisionTree, RandomForest

def train_val_split(x, y, val_size=0.2):
    total_samples = x.shape[0]
    idx = np.arange(total_samples)
    np.random.shuffle(idx)
    
    val_samples = int(total_samples * val_size)
    val_idx, train_idx = idx[:val_samples], idx[val_samples:]
    
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]

def test_model(model, x_train, y_train, x_val, y_val):
    start_time = datetime.datetime.now()

    if hasattr(model, "early_stopping") and model.early_stopping:
        model.fit(x_train, y_train, x_val, y_val, early_stopping=True)
    else:
        model.fit(x_train, y_train, x_val, y_val)

    y_pred = model.predict(x_val)
    accuracy = np.mean(y_val == y_pred)
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds() * 1000

    return accuracy, elapsed_time

def normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def load_dataset(dataset):
    if dataset == "iris":
        data = "data/iris.data"
        df = pd.read_csv(data, header=None)
        x = df.iloc[:, [0, 2]].values
        y = np.where(df.iloc[:, 4].values == 'Iris-setosa', 1, 0)
    elif dataset == "wdbc":
        data = "data/wdbc.data"
        df = pd.read_csv(data, header=None)
        df.drop(columns=[0], inplace=True)
        df[1] = df[1].map({'M': 1, 'B': 0})
        x = df.iloc[:, 1:].values
        y = df[1].values

    return normalize(x), y

def main():
    x, y = load_dataset("wdbc")
    x_train, y_train, x_val, y_val = train_val_split(x, y)

    models = [
        Perceptron(), AdalineGD(), AdalineSGD(), LogisticRegressionGD(), 
        DecisionTree(), RandomForest(n_trees=5, max_depth=1), 
        KNearestNeighbors(k=10)
    ]

    for model in models:
        accuracy, milliseconds = test_model(model, x_train, y_train, x_val, y_val)
        print(f"{model.__class__.__name__} fitting time: {milliseconds:.2f} ms")
        print(f"{model.__class__.__name__} accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()