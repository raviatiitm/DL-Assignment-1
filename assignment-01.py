import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.datasets import fashion_mnist
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import seaborn as sns

import wandb

# !wandb login 38f6722f118cc93758a91099d1f4a27a0c68274e

wandb.init(project="DL_Assignment1", entity="cs22m069")

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train = x_train / 255
x_test = x_test / 255

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

images = []
labels = []

n = len(x_train)

for i in range(n):
    if len(labels) == 10:
        break
    if classes[y_train[i]] not in labels:
        images.append(x_train[i])
        labels.append(classes[y_train[i]])

wandb.log(
    {"Question 1": [wandb.Image(img, caption=lbl) for img, lbl in zip(images, labels)]}
)
