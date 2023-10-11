import numpy as np
import arff

from models.neural_net import NeuralNetMLP

def load_arff_data(file_path):
    print("Loading data...")
    data = arff.load(open(file_path, 'r'))
    raw_data = np.array(data['data'])
    
    X = raw_data[:, :-1].astype(np.float32)
    y = raw_data[:, -1].astype(np.int64)
    
    return X, y

def train_test_split_manual(X, y, test_size=0.3, random_seed=123):
    rng = np.random.default_rng(random_seed)
    indices = rng.permutation(len(X))
    test_size = int(len(X) * test_size)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def train_neural_net_mlp(X_train, y_train, num_epochs=100, learning_rate=0.01, batch_size=64):
    num_features = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    num_hidden = 100
    
    model = NeuralNetMLP(num_features, num_hidden, num_classes)
    num_batches = len(X_train) // batch_size
    
    for epoch in range(num_epochs):
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]
            
            a_h, a_out = model.forward(X_batch)
            d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h = model.backward(X_batch, a_h, a_out, y_batch)
            
            model.weight_out -= learning_rate * d_loss__dw_out
            model.bias_out -= learning_rate * d_loss__db_out
            model.weight_h -= learning_rate * d_loss__d_w_h
            model.bias_h -= learning_rate * d_loss__d_b_h
        
        if epoch % 10 == 0:
            print("Epoch:", epoch)
        
    return model

def evaluate_model(model, X_test, y_test):
    _, predictions = model.forward(X_test)
    y_pred = np.argmax(predictions, axis=1)
    
    corrects = np.sum(y_pred == y_test)
    accuracy = corrects / len(y_test)
    
    print(f"Accuracy: {accuracy:.2%}")

def main():
    X, y = load_arff_data("data/mnist_784.arff")
    
    X_train, X_test, y_train, y_test = train_test_split_manual(X, y)
    
    model = train_neural_net_mlp(X_train, y_train)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()