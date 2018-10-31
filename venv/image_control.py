import cv2
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

ROW = 28; COLUMN = 28

def create_image_matrix(img_path):
    image = cv2.imread(img_path, 0) # read image as grayscale
    resized_image = cv2.resize(image, (ROW, COLUMN))
    x = []
    for row in resized_image:
        for pixel in row:
            x.append((pixel - 255)*(-1))
    return np.asarray(x)

image_data = create_image_matrix("images/1.png")
data = pd.read_csv("Data/train.csv").values

clf = DecisionTreeClassifier()

train_data = data[0:42000, 1:]
train_target = data[0:42000, 0]
clf.fit(train_data, train_target)

print("predicted: " + str(clf.predict([image_data])))