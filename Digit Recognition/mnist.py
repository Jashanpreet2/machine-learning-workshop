from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import csv 
import numpy as np
import pathlib
import os

def get_data(csv_path):
    X = []
    y = []

    with open(csv_path) as f:
        reader = csv.reader(f, delimiter=',', quotechar='|')
        for row in reader:
            row = [int(num) for num in row]
            y.append(row[0])
            x = np.array(row[1:])/255
            X.append(x)
    
    return X, y
    
    
train_X, train_y,  = get_data(os.path.join(pathlib.Path(__file__).parent.resolve(), "./MNIST_Dataset/mnist_train.csv"))
test_X, test_y = get_data(os.path.join(pathlib.Path(__file__).parent.resolve(), "./q3/MNIST_Dataset/mnist_test.csv"))

model = LogisticRegression(max_iter=1000)
model.fit(train_X, train_y)
test_predictions = model.predict(test_X)
print("Accuracy:", accuracy_score(test_predictions, test_y))
