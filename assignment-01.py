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
    def __init__(
        self,
        x_train,
        y_train,
        input_dim,
        hidden_layers_size,
        hidden_layers,
        output_dim,
        batch_size=32,
        epochs=1,
        activation_func="sigmoid",
        learning_rate=6e-3,
        decay_rate=0.9,
        beta=0.9,
        beta1=0.9,
        beta2=0.99,
        optimizer="rmsprop",
        weight_init="random",
    ):
        self.x_train, self.x_cv, self.y_train, self.y_cv = train_test_split(
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

        if optimizer == "gradient_descent":
            self.gradient_descent(self.x_train, self.y_train)
        elif optimizer == "sgd":
            self.sgd(self.x_train, self.y_train)
        elif optimizer == "nesterov":
            self.nesterov(self.x_train, self.y_train)
        elif optimizer == "adam":
            self.adam(self.x_train, self.y_train)
        elif optimizer == "nadam":
            self.nadam(self.x_train, self.y_train)
        elif optimizer == "momentum":
            self.momentum(self.x_train, self.y_train)
        elif optimizer == "rmsprop":
            self.rmsprop(self.x_train, self.y_train)

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

    def sgd(self, x_train, y_train):
        t = self.epochs
        for i in range(t):
            print("Epoch---", i + 1, end=" ")
            loss = 0
            val_loss = 0

            index = 1
            for x, y in zip(x_train, y_train):
                x = x.ravel()
                val = self.forward_propagation(x, y, self.weights, self.biases)
                loss += val
                self.backward_propagation(x, y, self.weights, self.biases)
                temp = index % self.batch
                if temp == 0 or index == x_train.shape[0]:
                    for j in range(len(self.weights)):
                        temp = self.learning_rate * self.weights_gradients[j]
                        self.weights[j] -= temp
                        self.biases[j] -= self.learning_rate * self.biases_gradients[j]
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
            cal_acc = self.calculate_accuracy(x_train, y_train)
            acc = round(cal_acc, 3)
            cal_acc_cv = self.calculate_accuracy(self.x_cv, self.y_cv)
            val_acc = round(cal_acc_cv, 3)
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

    def momentum(self, x_train, y_train):
        prev_gradients_w = []
        temp1 = []
        for i in self.weights_gradients:
            temp1.append(0 * i)
        prev_gradients_w = temp1
        prev_gradients_b = []
        temp2 = []
        for i in self.biases_gradients:
            temp2.append(0 * i)
        prev_gradients_b = temp2
        n = self.epochs

        for i in range(n):
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
                    for j in range(len(self.weights)):
                        v_w = (
                            self.decay_rate * prev_gradients_w[j]
                            + self.learning_rate * self.weights_gradients[j]
                        )
                        v_b = (
                            self.decay_rate * prev_gradients_b[j]
                            + self.learning_rate * self.biases_gradients[j]
                        )
                        self.weights[j] -= v_w
                        self.biases[j] -= v_b
                        prev_gradients_w[j] = v_w
                        prev_gradients_b[j] = v_b
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
                val = self.forward_propagation(x, y, self.weights, self.biases)
                val_loss += val

            cal_acc = self.calculate_accuracy(x_train, y_train)
            acc = round(cal_acc, 3)
            cal_acc_cv = self.calculate_accuracy(self.x_cv, self.y_cv)
            val_acc = round(cal_acc_cv, 3)
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

    def nesterov(self, x_train, y_train):
        prev_gradients_w = []
        temp1 = []
        for i in self.weights_gradients:
            temp1.append(0 * i)
        prev_gradients_w = temp1
        prev_gradients_b = []
        temp2 = []
        for i in self.biases_gradients:
            temp2.append(0 * i)
        prev_gradients_b = temp2

        n = self.epochs
        for i in range(n):
            print("Epoch---", i + 1, end=" ")
            loss = 0
            val_loss = 0
            weights = [
                self.weights[j] - self.decay_rate * prev_gradients_w[j]
                for j in range(len(self.weights))
            ]
            biases = [
                self.biases[j] - self.decay_rate * prev_gradients_b[j]
                for j in range(len(self.biases))
            ]
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
                    for j in range(len(self.weights)):
                        temp1 = (
                            self.decay_rate * prev_gradients_w[j]
                            + self.learning_rate * self.weights_gradients[j]
                        )
                        prev_gradients_w[j] = temp1
                        temp2 = (
                            self.decay_rate * prev_gradients_b[j]
                            + self.learning_rate * self.biases_gradients[j]
                        )
                        prev_gradients_b[j] = temp2

                        self.weights[j] -= prev_gradients_w[j]
                        self.biases[j] -= prev_gradients_b[j]

                    weights = [
                        self.weights[j] - self.decay_rate * prev_gradients_w[j]
                        for j in range(len(self.weights))
                    ]
                    biases = [
                        self.biases[j] - self.decay_rate * prev_gradients_b[j]
                        for j in range(len(self.biases))
                    ]
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
                val = self.forward_propagation(x, y, self.weights, self.biases)
                val_loss += val

            cal_acc = self.calculate_accuracy(x_train, y_train)
            acc = round(cal_acc, 3)
            cal_acc_cv = self.calculate_accuracy(self.x_cv, self.y_cv)
            val_acc = round(cal_acc_cv, 3)
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

    def rmsprop(self, x_train, y_train):
        prev_gradients_w = []
        temp1 = []
        for i in self.weights_gradients:
            temp1.append(0 * i)
        prev_gradients_w = temp1
        prev_gradients_b = []
        temp2 = []
        for i in self.biases_gradients:
            temp2.append(0 * i)
        prev_gradients_b = temp2

        eps = 1e-2
        n = self.epochs
        for i in range(n):
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
                condt = index % self.batch
                if condt == 0 or index == x_train.shape[0]:
                    for j in range(len(self.weights)):
                        t1 = (1 - self.beta) * np.square(self.weights_gradients[j])
                        v_w = self.beta * prev_gradients_w[j] + t1
                        t2 = (1 - self.beta) * np.square(self.biases_gradients[j])
                        v_b = self.beta * prev_gradients_b[j] + t2
                        denom_w = self.weights_gradients[j] / (np.sqrt(v_w + eps))
                        self.weights[j] -= self.learning_rate * denom_w
                        denom_b = self.biases_gradients[j] / (np.sqrt(v_b + eps))
                        self.biases[j] -= self.learning_rate * denom_b
                        prev_gradients_w[j] = v_w
                        prev_gradients_b[j] = v_b

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
                val = self.forward_propagation(x, y, self.weights, self.biases)
                val_loss += val

            cal_acc = self.calculate_accuracy(x_train, y_train)
            acc = round(cal_acc, 3)
            cal_acc_cv = self.calculate_accuracy(self.x_cv, self.y_cv)
            val_acc = round(cal_acc_cv, 3)
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

    def adam(self, x_train, y_train):
        m_prev_gradients_w = [0 * i for i in (self.weights_gradients)]
        m_prev_gradients_b = [0 * i for i in (self.biases_gradients)]
        v_prev_gradients_w = [0 * i for i in (self.weights_gradients)]
        v_prev_gradients_b = [0 * i for i in (self.biases_gradients)]

        iter = 1

        for i in range(self.epochs):
            print("Epoch---", i + 1, end=" ")
            loss = 0
            val_loss = 0
            eps = 1e-2
            self.weights_gradients = [0 * i for i in (self.weights_gradients)]
            self.biases_gradients = [0 * i for i in (self.biases_gradients)]

            index = 1
            for x, y in zip(x_train, y_train):
                x = x.ravel()
                loss += self.forward_propagation(x, y, self.weights, self.biases)
                self.backward_propagation(x, y, self.weights, self.biases)
                if index % self.batch == 0 or index == x_train.shape[0]:
                    for j in range(len(self.weights)):
                        m_w = (
                            self.beta1 * m_prev_gradients_w[j]
                            + (1 - self.beta1) * self.weights_gradients[j]
                        )
                        m_b = (
                            self.beta1 * m_prev_gradients_b[j]
                            + (1 - self.beta1) * self.biases_gradients[j]
                        )
                        v_w = self.beta2 * v_prev_gradients_w[j] + (
                            1 - self.beta2
                        ) * np.square(self.weights_gradients[j])
                        v_b = self.beta2 * v_prev_gradients_b[j] + (
                            1 - self.beta2
                        ) * np.square(self.biases_gradients[j])

                        m_hat_w = (m_w) / (1 - (self.beta1) ** iter)
                        m_hat_b = (m_b) / (1 - (self.beta1) ** iter)

                        v_hat_w = (v_w) / (1 - (self.beta2) ** iter)
                        v_hat_b = (v_b) / (1 - (self.beta2) ** iter)

                        self.weights[j] -= self.learning_rate * (
                            m_hat_w / (np.sqrt(v_hat_w + eps))
                        )
                        self.biases[j] -= self.learning_rate * (
                            m_hat_b / (np.sqrt(v_hat_b + eps))
                        )

                        m_prev_gradients_w[j] = m_w
                        m_prev_gradients_b[j] = m_b
                        v_prev_gradients_w[j] = v_w
                        v_prev_gradients_b[j] = v_b

                    self.weights_gradients = [0 * i for i in (self.weights_gradients)]
                    self.biases_gradients = [0 * i for i in (self.biases_gradients)]
                    iter += 1

                index += 1

            for x, y in zip(self.x_cv, self.y_cv):
                x = x.ravel()
                val_loss += self.forward_propagation(x, y, self.weights, self.biases)

            acc = round(self.calculate_accuracy(x_train, y_train), 3)
            val_acc = round(self.calculate_accuracy(self.x_cv, self.y_cv), 3)
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

    def nadam(self, x_train, y_train):
        m_prev_gradients_w = [0 * i for i in (self.weights_gradients)]
        m_prev_gradients_b = [0 * i for i in (self.biases_gradients)]
        v_prev_gradients_w = [0 * i for i in (self.weights_gradients)]
        v_prev_gradients_b = [0 * i for i in (self.biases_gradients)]

        iter = 1

        for i in range(self.epochs):
            print("Epoch---", i + 1, end=" ")
            loss = 0
            val_loss = 0
            eps = 1e-2
            self.weights_gradients = [0 * i for i in (self.weights_gradients)]
            self.biases_gradients = [0 * i for i in (self.biases_gradients)]

            index = 1
            for x, y in zip(x_train, y_train):
                x = x.ravel()
                loss += self.forward_propagation(x, y, self.weights, self.biases)
                self.backward_propagation(x, y, self.weights, self.biases)
                if index % self.batch == 0 or index == x_train.shape[0]:
                    for j in range(len(self.weights)):
                        m_w = (
                            self.beta1 * m_prev_gradients_w[j]
                            + (1 - self.beta1) * self.weights_gradients[j]
                        )
                        m_b = (
                            self.beta1 * m_prev_gradients_b[j]
                            + (1 - self.beta1) * self.biases_gradients[j]
                        )
                        v_w = self.beta2 * v_prev_gradients_w[j] + (
                            1 - self.beta2
                        ) * np.square(self.weights_gradients[j])
                        v_b = self.beta2 * v_prev_gradients_b[j] + (
                            1 - self.beta2
                        ) * np.square(self.biases_gradients[j])

                        m_hat_w = (m_w) / (1 - (self.beta1) ** iter)
                        m_hat_b = (m_b) / (1 - (self.beta1) ** iter)

                        v_hat_w = (v_w) / (1 - (self.beta2) ** iter)
                        v_hat_b = (v_b) / (1 - (self.beta2) ** iter)

                        m_dash_w = (
                            self.beta1 * m_hat_w
                            + (1 - self.beta1) * self.weights_gradients[j]
                        )
                        m_dash_b = (
                            self.beta1 * m_hat_b
                            + (1 - self.beta1) * self.biases_gradients[j]
                        )

                        self.weights[j] -= self.learning_rate * (
                            m_dash_w / (np.sqrt(v_hat_w + eps))
                        )
                        self.biases[j] -= self.learning_rate * (
                            m_dash_b / (np.sqrt(v_hat_b + eps))
                        )

                        m_prev_gradients_w[j] = m_w
                        m_prev_gradients_b[j] = m_b
                        v_prev_gradients_w[j] = v_w
                        v_prev_gradients_b[j] = v_b

                    self.weights_gradients = [0 * i for i in (self.weights_gradients)]
                    self.biases_gradients = [0 * i for i in (self.biases_gradients)]
                    iter += 1

                index += 1

            for x, y in zip(self.x_cv, self.y_cv):
                x = x.ravel()
                val_loss += self.forward_propagation(x, y, self.weights, self.biases)

            acc = round(self.calculate_accuracy(x_train, y_train), 3)
            val_acc = round(self.calculate_accuracy(self.x_cv, self.y_cv), 3)
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
        count = 0
        for i in range(len(X)):
            if self.predict(X[i]) == Y[i]:
                count += 1
        return count / len(X)

    def predict(self, x):
        x = x.ravel()
        self.activations[0] = x
        n = len(self.layers)
        for i in range(n - 2):
            if self.activation_func == "sigmoid":
                self.activations[i + 1] = self.sigmoid(
                    np.matmul(self.weights[i].T, self.activations[i]) + self.biases[i]
                )
            elif self.activation_func == "tanh":
                self.activations[i + 1] = self.tanh(
                    np.matmul(self.weights[i].T, self.activations[i]) + self.biases[i]
                )
            elif self.activation_func == "relu":
                self.activations[i + 1] = self.relu(
                    np.matmul(self.weights[i].T, self.activations[i]) + self.biases[i]
                )

        self.activations[n - 1] = self.softmax(
            np.matmul(self.weights[n - 2].T, self.activations[n - 2])
            + self.biases[n - 2]
        )

        return np.argmax(self.activations[-1])


sweep_config = {
    "method": "random",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "epochs": {"values": [3, 5, 10]},
        "layers": {"values": [[784, 80, 48, 16, 10], [784, 128, 64, 10]]},
        "learning_rate": {"values": [1e-3, 1e-4]},
        "optimizer": {"values": [SGD, NAG, SMGD, RMSPROP, ADAM, NADAM]},
        "batch_size": {"values": [16, 32, 64]},
        "initialisation": {"values": [RANDOM, XAVIER]},
        "activation": {"values": [SIGMOID, TAN_H, RELU]},
    },
}


def train_n_test():
    default_config = {
        "epochs": 3,
        "layers": [784, 128, 64, 10],
        "learning_rate": 1e-4,
        "optimizer": ADAM,
        "batch_size": 32,
        "initialisation": XAVIER,
        "activation": RELU,
    }

    wandb.init(
        config=default_config,
        project="Assignment 1",
        entity="iitm-cs6910-jan-may-2021-cs20m059-cs20m007",
    )
    config = wandb.config

    prefix = "Training"
    # Training
    if config.optimizer == SGD:
        W, B, _ = sgd(
            train_x,
            train_y,
            config.activation,
            config.epochs,
            config.layers,
            config.learning_rate,
            config.initialisation,
            config.batch_size,
            prefix,
        )
    elif config.optimizer == NAG:
        W, B, _ = nag(
            train_x,
            train_y,
            config.activation,
            config.epochs,
            config.layers,
            config.learning_rate,
            config.initialisation,
            config.batch_size,
            prefix,
        )
    elif config.optimizer == SMGD:
        W, B, _ = stochastic_momentum(
            train_x,
            train_y,
            config.activation,
            config.epochs,
            config.layers,
            config.learning_rate,
            config.initialisation,
            config.batch_size,
            prefix,
        )
    elif config.optimizer == RMSPROP:
        W, B, _ = rmsprop(
            train_x,
            train_y,
            config.activation,
            config.epochs,
            config.layers,
            config.learning_rate,
            config.initialisation,
            config.batch_size,
            prefix,
        )
    elif config.optimizer == ADAM:
        W, B, _ = adam(
            train_x,
            train_y,
            config.activation,
            config.epochs,
            config.layers,
            config.learning_rate,
            config.initialisation,
            config.batch_size,
            prefix,
        )
    else:
        W, B, _ = nadam(
            train_x,
            train_y,
            config.activation,
            config.epochs,
            config.layers,
            config.learning_rate,
            config.initialisation,
            config.batch_size,
            prefix,
        )

    # Validation
    val_error_list = []
    validation_accuracy = 0
    for i in range(len(val_y)):
        H, A, y_hat = forward_propagation(
            test_x[i], config.activation, W, B, config.layers
        )
        error_val = cross_entropy_loss(val_y[i], y_hat)
        val_error_list.append(error_val)
        y = y_hat.copy()
        y = np.reshape(y, 10)
        pred = np.argmax(y, axis=0)
        if pred == val_y[i]:
            validation_accuracy += 1
        log_acc_val = float(validation_accuracy / (i + 1))
        log_error_val = float(np.average(val_error_list))
        wandb.log({"accuracy_val": log_acc_val, "loss_val": log_error_val})

    # Testing
    prediction_list = []
    error_list = []
    accuracy = 0
    for i in range(len(test_y)):
        H, A, y_hat = forward_propagation(
            test_x[i], config.activation, W, B, config.layers
        )
        error_val = cross_entropy_loss(test_y[i], y_hat)
        error_list.append(error_val)
        y = y_hat.copy()
        y = np.reshape(y, 10)
        pred = np.argmax(y, axis=0)
        prediction_list.append(pred)
        if pred == test_y[i]:
            accuracy += 1
        log_acc_val = float(accuracy / (i + 1))
        log_error_val = float(np.average(error_list))
        wandb.log({"accuracy": log_acc_val, "loss": log_error_val})

    wandb.log(
        {
            f"{prefix} Confusion Matrix": wandb.plot.confusion_matrix(
                probs=None, y_true=prediction_list, preds=test_y, class_names=classes
            )
        }
    )

    print("Done!")
