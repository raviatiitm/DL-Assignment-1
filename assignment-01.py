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


class Neural_network:
    np.random.seed(10)

    def __init__(
        self,
        x_train,
        y_train,
        input_dim,
        hidden_layers_size,
        hidden_layers,
        output_dim,
        batch_size=32,
        epochs=2,
        activation_func="sigmoid",
        learning_rate=6e-3,
        decay_rate=0.9,
        beta=0.9,
        beta1=0.9,
        beta2=0.99,
        optimizer="gradient_descent",
        weight_init="random",
    ):
        x_train, self.x_cv, y_train, self.y_cv = train_test_split(
            x_train, y_train, test_size=0.10, random_state=100, stratify=y_train
        )

        np.random.seed(10)
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.hidden_layers_size = hidden_layers_size
        self.output_dim = output_dim

        self.batch = batch_size
        self.epochs = epochs
        self.activation_func = activation_func
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.weight_init = weight_init
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2

        self.layers = [self.input_dim] + self.hidden_layers_size + [self.output_dim]

        layers = self.layers.copy()
        self.weights = []
        self.biases = []
        self.activations = []
        self.activation_gradients = []
        self.weights_gradients = []
        self.biases_gradients = []
        n = len(layers)
        for i in range(n - 1):
            if self.weight_init == "random":
                a = np.random.normal(0, 0.5, (layers[i], layers[i + 1]))
                self.weights.append(a)
                self.biases.append(np.random.normal(0, 0.5, (layers[i + 1])))
            else:
                std = np.sqrt(2 / (layers[i] * layers[i + 1]))
                a = np.random.normal(0, std, (layers[i], layers[i + 1]))
                self.weights.append(a)
                self.biases.append(np.random.normal(0, std, (layers[i + 1])))
            v1 = np.zeros(layers[i])
            self.activations.append(v1)
            v2 = np.zeros(layers[i + 1])
            self.activation_gradients.append(v2)
            self.weights_gradients.append(np.zeros((layers[i], layers[i + 1])))
            self.biases_gradients.append(v2)

        self.activations.append(np.zeros(layers[-1]))

        self.gradient_descent(x_train, y_train)

    def sigmoid(self, activations):
        res = []
        for z in activations:
            if z < -40:
                res.append(0.0)
            elif z > 40:
                res.append(1.0)
            else:
                res.append(1 / (1 + np.exp(-z)))
        res = np.asarray(res)
        return res

    def tanh(self, activations):
        res = []
        for z in activations:
            if z < -20:
                res.append(-1.0)
            elif z > 20:
                res.append(1.0)
            else:
                temp = (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
                res.append(temp)
        res = np.asarray(res)
        return res

    def relu(self, activations):
        res = []
        for i in activations:
            if i > 0:
                res.append(i)
            else:
                res.append(0)
        res = np.asarray(res)
        return res

    def softmax(self, activations):
        tot = 0
        res = []
        for z in activations:
            tot += np.exp(z)
        res = np.asarray([np.exp(z) / tot for z in activations])
        return res

    def forward_propagation(self, x, y, weights, biases):
        self.activations[0] = x
        n = len(self.layers)
        for i in range(n - 2):
            if self.activation_func == "sigmoid":
                s = self.sigmoid(
                    np.matmul(weights[i].T, self.activations[i]) + biases[i]
                )
                self.activations[i + 1] = s
            elif self.activation_func == "tanh":
                t = self.tanh(np.matmul(weights[i].T, self.activations[i]) + biases[i])
                self.activations[i + 1] = t
            elif self.activation_func == "relu":
                r = self.relu(np.matmul(weights[i].T, self.activations[i]) + biases[i])
                self.activations[i + 1] = r
        temp = self.softmax(
            np.matmul(weights[n - 2].T, self.activations[n - 2]) + biases[n - 2]
        )
        self.activations[n - 1] = temp
        return -(
            np.log2(self.activations[-1][y])
        )  # Return cross entropy loss for single data point.

    def grad_w(self, i):
        gw = np.matmul(
            self.activations[i].reshape((-1, 1)),
            self.activation_gradients[i].reshape((1, -1)),
        )
        return gw

    def grad_b(self, i):
        gb = self.activation_gradients[i]
        return gb

    def backward_propagation(self, x, y, weights, biases):
        y_onehot = np.zeros(self.output_dim)
        y_onehot[y] = 1
        self.activation_gradients[-1] = -1 * (y_onehot - self.activations[-1])
        n = len(self.layers)
        for i in range(n - 2, -1, -1):
            gw = self.grad_w(i)
            self.weights_gradients[i] += gw
            gb = self.grad_b(i)
            self.biases_gradients[i] += gb
            if i != 0:
                value = np.matmul(weights[i], self.activation_gradients[i])
                if self.activation_func == "sigmoid":
                    val = value * self.activations[i] * (1 - self.activations[i])
                    self.activation_gradients[i - 1] = val
                elif self.activation_func == "tanh":
                    val = value * (1 - np.square(self.activations[i]))
                    self.activation_gradients[i - 1] = val
                elif self.activation_func == "relu":
                    res = []
                    for k in self.activations[i]:
                        if k <= 0:
                            res.append(0.0)
                        else:
                            res.append(1.0)
                    res = np.asarray(res)
                    self.activation_gradients[i - 1] = value * res

    def gradient_descent(self, x_train, y_train):
        for i in range(self.epochs):
            print("Epoch---", i + 1, end=" ")
            loss = 0
            val_loss = 0
            wg = []
            for i in self.weights_gradients:
                wg.append(0 * i)
            self.weights_gradients = wg
            bg = []
            for i in self.biases_gradients:
                bg.append(0 * i)
            self.biases_gradients = bg

            index = 1
            for x, y in zip(x_train, y_train):
                x = x.ravel()
                val = self.forward_propagation(x, y, self.weights, self.biases)
                loss += val
                self.backward_propagation(x, y, self.weights, self.biases)
                temp = index % self.batch
                if temp == 0 or index == x_train.shape[0]:
                    n = len(self.weights)
                    for j in range(n):
                        w_g = self.learning_rate * self.weights_gradients[j]
                        self.weights[j] -= w_g
                        b_g = self.learning_rate * self.biases_gradients[j]
                        self.biases[j] -= b_g
                    wg = []
                    for i in self.weights_gradients:
                        wg.append(0 * i)
                    self.weights_gradients = wg
                    bg = []
                    for i in self.biases_gradients:
                        bg.append(0 * i)
                    self.biases_gradients = bg
                index += 1

            for x, y in zip(self.x_cv, self.y_cv):
                x = x.ravel()
                temp = self.forward_propagation(x, y, self.weights, self.biases)
                val_loss += temp
            temp1 = self.calculate_accuracy(x_train, y_train)
            acc = round(temp1, 3)
            temp2 = self.calculate_accuracy(self.x_cv, self.y_cv)
            val_acc = round(temp2, 3)
            print(
                "  loss = ",
                loss / x_train.shape[0],
                "  accuracy = ",
                acc,
                "   validation loss= ",
                val_loss / self.x_cv.shape[0],
                "  validation accuaracy= ",
                val_acc,
            )

    def calculate_accuracy(self, X, Y):
        n = len(X)
        count = 0
        for i in range(n):
            if self.predict(X[i]) == Y[i]:
                count += 1
        res = count / n
        return res

    def predict(self, x):
        n = len(self.layers)
        res = []
        x = x.ravel()
        self.activations[0] = x
        for i in range(n - 2):
            if self.activation_func == "sigmoid":
                sg_val = self.sigmoid(
                    np.matmul(self.weights[i].T, self.activations[i]) + self.biases[i]
                )
                self.activations[i + 1] = sg_val
            elif self.activation_func == "tanh":
                th_val = self.tanh(
                    np.matmul(self.weights[i].T, self.activations[i]) + self.biases[i]
                )
                self.activations[i + 1] = th_val
            elif self.activation_func == "relu":
                r_val = self.relu(
                    np.matmul(self.weights[i].T, self.activations[i]) + self.biases[i]
                )
                self.activations[i + 1] = r_val
        val = self.softmax(
            np.matmul(self.weights[n - 2].T, self.activations[n - 2])
            + self.biases[n - 2]
        )
        self.activations[n - 1] = val

        return np.argmax(self.activations[-1])
