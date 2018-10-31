import numpy as np
import matplotlib.pyplot as pt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("Data/train.csv").values
clf = DecisionTreeClassifier()

# Define Train and Data
train_test_ratio = int(input("Please enter the ratio of train data to test data: "))
last_train = 42000/(train_test_ratio+1)*train_test_ratio
first_test = last_train + 1

# Train Data
train_data = data[0:int(last_train), 1:]
train_target = data[0:int(last_train), 0]

# Test Data
test_data = data[int(first_test):42000, 1:]
test_target = data[int(first_test):42000, 0]

# Train
clf.fit(train_data, train_target)

# Test
predicts = clf.predict(test_data)
count = 0
for i in range(0, len(test_target)-1):
    count +=1 if predicts[i] == test_target[i] else 0
print("Accuracy = %" + str(count/len(test_target)*100))