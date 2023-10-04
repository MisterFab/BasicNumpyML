import pandas as pd
import numpy as np
import datetime

from classification import Perceptron, AdalineGD, AdalineSGD, LogisticRegressionGD

def train_val_split(x, y, val_size=0.2):
    total_samples = x.shape[0]
    idx = np.arange(total_samples)
    np.random.shuffle(idx)
    
    val_samples = int(total_samples * val_size)
    
    val_idx = idx[:val_samples]
    train_idx = idx[val_samples:]
    
    x_train, x_val = x[train_idx], x[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    return x_train, y_train, x_val, y_val

def test_model(model, x_train, y_train, x_val, y_val):
    time = datetime.datetime.now()
    model.fit(x_train, y_train, x_val, y_val, early_stopping=True)
    y_pred = model.predict(x_val)
    correct_predictions = y_val == y_pred
    accuracy = np.mean(correct_predictions)
    elapsed_time = datetime.datetime.now() - time
    milliseconds = elapsed_time.total_seconds() * 1000
    return accuracy, milliseconds

def normalize(data):
    mean = np.mean(data, axis=0)
    std_dev = np.std(data, axis=0)
    return (data - mean) / std_dev

def iris():
    data = "data/iris.data"
    df = pd.read_csv(data, header=None)

    x = df.iloc[:, [0, 2]].values
    y = df.iloc[:, 4].values
    y = np.where(y == 'Iris-setosa', 1, 0)

    x = normalize(x)
    return x, y

def wdbc():
    data = "data/wdbc.data"
    df = pd.read_csv(data, header=None)

    df.drop(columns=[0], inplace=True)
    df[1] = df[1].map({'M': 1, 'B': 0})
    x = df.iloc[:, 1:].values
    y = df[1].values

    x = normalize(x)
    return x, y

def main():
    x, y = wdbc()

    x_train, y_train, x_val, y_val = train_val_split(x, y)

    models = [Perceptron(), AdalineGD(), AdalineSGD(), LogisticRegressionGD()]

    for model in models:
        accuracy, milliseconds = test_model(model, x_train, y_train, x_val, y_val) # Updated to pass validation data
        print(f"{model.__class__.__name__} fitting time: {milliseconds:.2f} ms")
        print(f"{model.__class__.__name__} accuracy: {accuracy:.2%}")


if __name__ == "__main__":
    main()
