import pandas as pd
import numpy as np
import datetime
from models.classification import (Perceptron, AdalineGD, AdalineSGD, 
                                  LogisticRegressionGD, KNearestNeighbors)
from models.tree import DecisionTree, RandomForest
from models.regression import LinearRegressionGD, RANSACRegressor

def train_val_split(x, y, val_size=0.2):
    val_samples = int(len(x) * val_size)
    idx = np.random.permutation(len(x))
    return x[idx[val_samples:]], y[idx[val_samples:]], x[idx[:val_samples]], y[idx[:val_samples]]

def test_model(model, x_train, y_train, x_val, y_val):
    start_time = datetime.datetime.now()

    if hasattr(model, "early_stopping") and model.early_stopping:
        model.fit(x_train, y_train, x_val, y_val, early_stopping=True)
    elif isinstance(model, RANSACRegressor):
        model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train, x_val, y_val)

    if isinstance(model, (LinearRegressionGD, RANSACRegressor)):
        metric = np.sqrt(np.mean((y_val - model.predict(x_val))**2))
        metric_name = "RMSE"
    else:
        metric = np.mean(y_val == model.predict(x_val))
        metric_name = "accuracy"

    elapsed_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
    return metric_name, metric, elapsed_time

def normalize(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def load_dataset(dataset):
    data_loaders = {
        "iris": load_iris,
        "wdbc": load_wdbc,
        "realestate": load_realestate
    }
    return data_loaders[dataset]()

def load_iris():
    df = pd.read_csv("data/iris.data", header=None)
    x = df.iloc[:, [0, 2]].values
    y = np.where(df.iloc[:, 4].values == 'Iris-setosa', 1, 0)
    return normalize(x), y

def load_wdbc():
    df = pd.read_csv("data/wdbc.data", header=None).drop(columns=[0])
    df[1] = df[1].map({'M': 1, 'B': 0})
    return normalize(df.iloc[:, 1:].values), df[1].values

def load_realestate():
    df = pd.read_csv("data/real_estate.csv").drop(columns=["No"])
    return normalize(df.iloc[:, :-1].values), df.iloc[:, -1].values

def display_results(model, metric_name, metric, elapsed_time):
    print(f"{model.__class__.__name__} fitting time: {elapsed_time:.2f} ms")
    if metric_name == "accuracy":
        print(f"{model.__class__.__name__} {metric_name}: {metric:.2%}")
    else:
        print(f"{model.__class__.__name__} {metric_name}: {metric:.4f}")

def main():
    for dataset_name, models in [("wdbc", [Perceptron(), AdalineGD(), AdalineSGD(), LogisticRegressionGD(), DecisionTree(), RandomForest(n_trees=5, max_depth=1), KNearestNeighbors(k=10)]),
                                 ("realestate", [LinearRegressionGD(), RANSACRegressor(LinearRegressionGD())])]:
        x, y = load_dataset(dataset_name)
        x_train, y_train, x_val, y_val = train_val_split(x, y)

        for model in models:
            metric_name, metric, elapsed_time = test_model(model, x_train, y_train, x_val, y_val)
            display_results(model, metric_name, metric, elapsed_time)

if __name__ == "__main__":
    main()