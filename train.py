

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import fashion_mnist,mnist
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import seaborn as sns
import wandb

parser = argparse.ArgumentParser(description='Run my model function')
parser.add_argument('-wp','--wandb_project', default="DL_Assignment1", required=False,metavar="", type=str, help=' ')
parser.add_argument('-we','--wandb_entity', default="cs22m069", required=False,metavar="", type=str, help='')
parser.add_argument('-d','--dataset', default="fashion_mnist", required=False,metavar="", type=str,choices= ["mnist", "fashion_mnist"], help=' ')
parser.add_argument('-e','--epochs', default=10, required=False,metavar="", type=int, help=' ')
parser.add_argument('-b','--batch_size', default=32, required=False,metavar="", type=int, help=' ')
parser.add_argument('-o','--optimizer', default="adam", required=False,metavar="", type=str,choices= ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help=' ')
parser.add_argument('-l','--loss', default="cross_entropy", required=False,metavar="", type=str,choices= ["mean_squared_error", "cross_entropy"], help=' ')
parser.add_argument('-lr','--learning_rate', default=0.01, required=False,metavar="", type=float, help='')
parser.add_argument('-m','--momemtum', default=0.9, required=False,metavar="", type=float, help=' ')
parser.add_argument('-beta','--beta', default=0.9, required=False,metavar="", type=float, help=' ')
parser.add_argument('-beta1','--beta1', default=0.9, required=False,metavar="", type=float, help=' ')
parser.add_argument('-beta2','--beta2', default=0.99, required=False,metavar="", type=float, help=' ')
parser.add_argument('-eps','--epsilon', default=0.00001, required=False,metavar="", type=float, help=' ')
parser.add_argument('-w_d','--weight_decay', default=.0, required=False,metavar="", type=float, help=' ')
parser.add_argument('-w_i','--weight_init', default="random", required=False,metavar="", type=str,choices= ["random", "Xavier"], help=' ')
parser.add_argument('-nhl','--num_layers', default=3, required=False,metavar="", type=int, help=' ')
parser.add_argument('-sz','--hidden_size', default=32, required=False,metavar="", type=int, help=' ')
parser.add_argument('-a','--activation', default="sigmoid", required=False,metavar="", type=str,choices= ["identity", "sigmoid", "tanh", "ReLU"], help=' ')
args = parser.parse_args()


if args.dataset=='fashion_mnist':
  (x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
else:
  (x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train = x_train/255
x_test = x_test/255

if args.optimizer=='nag':
  args.optimizer='nesterov'
if args.weight_init=='Xavier':
  args.weight_init='xavier'
if args.loss=='mean_squared_error':
  args.loss='mse'
if args.activation=='ReLU':
  args.activation='relu'


class Neural_network:
    def __init__(self,x_train,y_train,input_dim,hidden_layers_size,hidden_layers,output_dim,batch_size=32,epochs=1,               activation_func="sigmoid",learning_rate=6e-3 ,decay_rate=0.9,beta=0.9,beta1=0.9,beta2=0.99,optimizer="nesterov",weight_init="random",loss='cross_entropy'):

        self.x_train,self.x_cv,self.y_train,self.y_cv = train_test_split(x_train, y_train, test_size=0.10, random_state=100,stratify=y_train)

        np.random.seed(10)
        self.gradient={}
        for i in range(hidden_layers+2):
            self.gradient["W"+str(i)]=i;
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
        for i in range(hidden_layers+2):
            self.gradient["b"+str(i)]=i;
        self.weight_init = weight_init
        self.beta = beta
        self.beta1 = beta1
        self.beta2 = beta2
        self.layers = [self.input_dim] + self.hidden_layers*[self.hidden_layers_size] + [self.output_dim]
        layers = self.layers.copy()
        self.activations = []
        self.loss = loss
        self.activation_gradients = []
        self.optimizer_list={'gradient_descent':self.gradient_descent,'sgd':self.sgd,'nesterov':self.nesterov,'nadam':self.nadam,'adam':self.adam,'momentum':self.momentum,'rmsprop':self.rmsprop}
        self.weights_gradients = []
        self.biases_gradients = []
        self.weights = []
        self.biases = []
        n=len(layers)
        for i in range(n-1):
            if self.weight_init == 'random':
                a=np.random.normal(0,0.5,(layers[i],layers[i+1]))
                self.weights.append(a)
                self.biases.append(np.random.normal(0,0.5,(layers[i+1])))
            else :
                std = np.sqrt(2/(layers[i]*layers[i+1]))
                a=np.random.normal(0,std,(layers[i],layers[i+1]))
                self.weights.append(a)
                self.biases.append(np.random.normal(0,std,(layers[i+1])))
            v1=np.zeros(layers[i])
            self.activations.append(v1)
            v2=np.zeros(layers[i+1])
            self.activation_gradients.append(v2)
            self.weights_gradients.append(np.zeros((layers[i],layers[i+1])))
            self.biases_gradients.append(v2)
        self.activations.append(np.zeros(layers[-1]))
        self.optimizer_list[optimizer](self.x_train,self.y_train)
            

    def sigmoid(self,activations):
        res = []
        for z in activations:
            if z<-40:
                res.append(0.0)
            elif z>40:
                res.append(1.0)
            else:
                res.append(1/(1+np.exp(-z)))
        res=np.asarray(res)
        return res

    def tanh(self,activations):
        res = []
        for z in activations:
            if z<-20:
                res.append(-1.0)
            elif z>20:
                res.append(1.0)
            else:
                temp=(np.exp(z) - np.exp(-z))/(np.exp(z) + np.exp(-z))
                res.append(temp)
        res=np.asarray(res)
        return res

    def relu(self,activations):
        res = []
        for i in activations:
            if i>0:
                res.append(i)
            else:
                res.append(0)
        res=np.asarray(res)
        return res

    def softmax(self,activations):
        tot = 0
        res=[]
        for z in activations:
            tot += np.exp(z)
        res=np.asarray([np.exp(z)/tot for z in activations])
        return res
    

    def forward_propagation(self,x,y,weights,biases):
        n = len(self.layers)
        pre_activation=[]
        for i in range(n-2):
            pre_activation.append(i)
        self.activations[0] = x
        for i in range(n-2):
            if self.activation_func == "sigmoid":
                s=self.sigmoid(np.matmul(weights[i].T,self.activations[i])+biases[i])
                self.activations[i+1] =s
            elif self.activation_func == "tanh":
                t=self.tanh(np.matmul(weights[i].T,self.activations[i])+biases[i])
                self.activations[i+1] =t
            elif self.activation_func == "relu":
                r=self.relu(np.matmul(weights[i].T,self.activations[i])+biases[i])
                self.activations[i+1] = r
        temp=self.softmax(np.matmul(weights[n-2].T,self.activations[n-2])+biases[n-2])
        self.activations[n-1] = temp      
        if self.loss == "cross_entropy":      
          return -(np.log2(self.activations[-1][y])) 
        elif self.loss == "mse":
          y_onehot = np.zeros(self.output_dim)
          y_onehot[y] = 1
          return np.sum(np.square(self.activations[-1] - y_onehot))


    def grad_w(self,i):
        gw=np.matmul(self.activations[i].reshape((-1,1)),self.activation_gradients[i].reshape((1,-1)))
        return gw


    def grad_b(self,i):
        gb=self.activation_gradients[i]
        return gb


    def backward_propagation(self,x,y,weights,biases):
        y_onehot = np.zeros(self.output_dim)
        y_onehot[y] = 1
        if self.loss == "cross_entropy": 
          self.activation_gradients[-1] =  -1*(y_onehot - self.activations[-1])
        elif self.loss == "mse":
          temp_vec = 2 * (self.activations[-1] - y_onehot) * (self.activations[-1])
          for i in range(len(self.activations[-1])):
            i_onehot=np.zeros(self.output_dim)
            i_onehot[i]=1
            self.activation_gradients[-1][i] = np.dot(temp_vec,(i_onehot - np.asarray([self.activations[-1][i]]*self.output_dim)))
        n = len(self.layers)
        for i in range(n-2,-1,-1):
            gw=self.grad_w(i)
            self.weights_gradients[i] += gw
            gb= self.grad_b(i)
            self.biases_gradients[i] +=gb
            if i!=0:
                val1=self.activation_gradients[i]
                value = np.matmul(weights[i],val1)
                if self.activation_func == "sigmoid":
                    val= value * self.activations[i] * (1-self.activations[i])
                    self.activation_gradients[i-1] = val
                elif self.activation_func == "tanh":
                    val=value * (1-np.square(self.activations[i]))
                    self.activation_gradients[i-1] = val
                elif self.activation_func == "relu":
                    res = []
                    for k in self.activations[i]:
                        ans=1.0 if k>0 else 0.0
                        res.append(ans)
                    res = np.asarray(res)
                    self.activation_gradients[i-1] = value * res
                   

    def gradient_descent(self,x_train,y_train):
        grads=[]
        for i in (self.weights_gradients):
            grads.append(i)
        for i in range(self.epochs):
            print('Epoch---',i+1,end=" ")
            loss = 0
            val_loss = 0
            wg=[]
            for i in (self.weights_gradients):
                wg.append(0*i)
            self.weights_gradients = wg
            bg=[]
            for i in (self.biases_gradients):
              bg.append(0*i)
            self.biases_gradients =bg
            index = 1
            for x,y in zip(x_train,y_train):
                x = x.ravel()
                val=self.forward_propagation(x,y,self.weights,self.biases)
                loss += val
                self.backward_propagation(x,y,self.weights,self.biases)
                temp=index % self.batch
                if temp == 0 or index == x_train.shape[0]:
                    n=len(self.weights)
                    for j in range(n):
                        w_g=self.learning_rate * self.weights_gradients[j]
                        self.weights[j] -= w_g
                        b_g=self.learning_rate * self.biases_gradients[j]
                        self.biases[j] -= b_g
                    wg=[]
                    for i in (self.weights_gradients):
                      wg.append(0*i)
                    self.weights_gradients = wg
                    bg=[]
                    for i in (self.biases_gradients):
                      bg.append(0*i)
                    self.biases_gradients =bg
                index += 1 
            for x,y in zip(self.x_cv,self.y_cv):
               x=x.ravel()
               temp=self.forward_propagation(x,y,self.weights,self.biases)
               val_loss+=temp
            temp1=self.calculate_accuracy(x_train,y_train)
            acc=round(temp1,3)
            temp2=self.calculate_accuracy(self.x_cv,self.y_cv)
            val_acc=round(temp2,3)
            print('  loss = ',loss/x_train.shape[0],'  accuracy = ',acc,'   validation loss= ',val_loss/self.x_cv.shape[0],'  validation accuaracy= ',val_acc)

    def sgd(self,x_train,y_train):
        grads=[]
        for i in (self.weights_gradients):
            grads.append(i)
        t=self.epochs
        for i in range(t):
            print('Epoch---',i+1,end=" ")
            loss = 0
            val_loss=0
            index = 1
            for x,y in zip(x_train,y_train):
                x = x.ravel()
                val=self.forward_propagation(x,y,self.weights,self.biases)
                loss += val
                self.backward_propagation(x,y,self.weights,self.biases)
                temp=index % self.batch
                if  temp== 0 or index == x_train.shape[0]:
                    lst=[0*i for i in (self.weights_gradients)]
                    for j in range(len(self.weights)):
                        temp=self.learning_rate * self.weights_gradients[j]
                        self.weights[j] -= temp
                        self.biases[j] -= self.learning_rate * self.biases_gradients[j]
                    wg=[]
                    for i in (self.weights_gradients):
                      wg.append(0*i)
                    self.weights_gradients = wg
                    bg=[]
                    for i in (self.biases_gradients):
                      bg.append(0*i)
                    self.biases_gradients =bg
                index +=1   
            for x,y in zip(self.x_cv,self.y_cv):
               x=x.ravel()
               temp=self.forward_propagation(x,y,self.weights,self.biases)
               val_loss+=temp
            cal_acc=self.calculate_accuracy(x_train,y_train)
            acc=round(cal_acc,3)
            cal_acc_cv=self.calculate_accuracy(self.x_cv,self.y_cv)
            val_acc=round(cal_acc_cv,3)
            wandb.log({'train_loss':loss/x_train.shape[0],'train_accuracy':acc,'val_loss':val_loss/self.x_cv.shape[0],'val_accuracy':val_acc})
            print('  loss = ',loss/x_train.shape[0],'  accuracy = ',acc,'   validation loss= '
                  ,val_loss/self.x_cv.shape[0],'  validation accuaracy= ',val_acc)
            
    def momentum(self,x_train,y_train):
        prev_gradients_w=[]
        temp1=[]
        for i in (self.weights_gradients):
            temp1.append(0*i)
        prev_gradients_w=temp1
        prev_gradients_b=[]
        temp2=[]
        for i in (self.biases_gradients):
            temp2.append(0*i)
        prev_gradients_b=temp2
        n=self.epochs

        for i in range(n):
            print('Epoch---',i+1,end=" ")
            loss = 0
            val_loss=0
            wg=[]
            for i in (self.weights_gradients):
              wg.append(0*i)
            self.weights_gradients = wg
            bg=[]
            for i in (self.biases_gradients):
              bg.append(0*i)
            self.biases_gradients=bg
            index = 1
            for x,y in zip(x_train,y_train):
                x = x.ravel()
                val=self.forward_propagation(x,y,self.weights,self.biases)
                loss += val
                self.backward_propagation(x,y,self.weights,self.biases)
                temp=index % self.batch
                if  temp== 0 or index == x_train.shape[0]:
                    lst=[0*i for i in (self.weights_gradients)]
                    for j in range(len(self.weights)):
                        v1=self.learning_rate * self.weights_gradients[j]
                        v_w =(self.decay_rate * prev_gradients_w[j] +v1)
                        v2= self.learning_rate * self.biases_gradients[j]
                        v_b = (self.decay_rate * prev_gradients_b[j] + v2)
                        self.weights[j] -= v_w
                        self.biases[j] -= v_b
                        prev_gradients_w[j] = v_w
                        prev_gradients_b[j] = v_b
                    wg=[]
                    for i in (self.weights_gradients):
                      wg.append(0*i)
                    self.weights_gradients = wg
                    bg=[]
                    for i in (self.biases_gradients):
                      bg.append(0*i)
                    self.biases_gradients=bg
                index +=1
            for x,y in zip(self.x_cv,self.y_cv):
               x=x.ravel()
               val=self.forward_propagation(x,y,self.weights,self.biases)
               val_loss+=val

            cal_acc=self.calculate_accuracy(x_train,y_train)
            acc=round(cal_acc,3)
            cal_acc_cv=self.calculate_accuracy(self.x_cv,self.y_cv)
            val_acc=round(cal_acc_cv,3)
            wandb.log({'train_loss':loss/x_train.shape[0],'train_accuracy':acc,'val_loss':val_loss/self.x_cv.shape[0],'val_accuracy':val_acc})
            print('  loss = ',loss/x_train.shape[0],'  accuracy = ',acc,'   validation loss= '
                  ,val_loss/self.x_cv.shape[0],'  validation accuaracy= ',val_acc)


    def nesterov(self,x_train,y_train):
        prev_gradients_w=[]
        temp1=[]
        for i in (self.weights_gradients):
            temp1.append(0*i)
        prev_gradients_w=temp1
        prev_gradients_b=[]
        temp2=[]
        for i in (self.biases_gradients):
            temp2.append(0*i)
        prev_gradients_b=temp2

        n=self.epochs
        for i in range(n):
            print('Epoch---',i+1,end=" ")
            loss = 0
            val_loss=0
            for j in range(len(self.weights)):
              temp=self.weights[j] -  (self.decay_rate * prev_gradients_w[j])
              self.weights[j]=temp
              self.biases[j] =self.biases[j] -  self.decay_rate * prev_gradients_b[j]
            wg=[]
            for i in (self.weights_gradients):
              wg.append(0*i)
            self.weights_gradients = wg
            bg=[]
            for i in (self.biases_gradients):
              bg.append(0*i)
            self.biases_gradients=bg
            index = 1
            for x,y in zip(x_train,y_train):
                x = x.ravel()
                val=self.forward_propagation(x,y,self.weights,self.biases)
                loss += val
                self.backward_propagation(x,y,self.weights,self.biases)
                temp=index % self.batch
                if temp == 0 or index == x_train.shape[0]:
                    lst=[0*i for i in (self.weights_gradients)]
                    for j in range(len(self.weights)):
                        temp1=self.decay_rate * prev_gradients_w[j] + self.learning_rate*self.weights_gradients[j]
                        prev_gradients_w[j] =temp1
                        temp2= self.decay_rate * prev_gradients_b[j] + self.learning_rate*self.biases_gradients[j]               
                        prev_gradients_b[j] =  temp2
                                        
                        self.weights[j] -= prev_gradients_w[j]
                        self.biases[j] -= prev_gradients_b[j]
                    weights = [self.weights[j] -  self.decay_rate * prev_gradients_w[j] for j in range(len(self.weights))]
                    biases = [self.biases[j] -  self.decay_rate * prev_gradients_b[j] for j in range(len(self.biases))]
                    wg=[]
                    for i in (self.weights_gradients):
                       wg.append(0*i)
                    self.weights_gradients = wg
                    bg=[]
                    for i in (self.biases_gradients):
                      bg.append(0*i)
                    self.biases_gradients=bg
                index += 1
            for x,y in zip(self.x_cv,self.y_cv):
               x=x.ravel()
               val=self.forward_propagation(x,y,self.weights,self.biases)
               val_loss+=val
            cal_acc=self.calculate_accuracy(x_train,y_train)
            acc=round(cal_acc,3)
            cal_acc_cv=self.calculate_accuracy(self.x_cv,self.y_cv)
            val_acc=round(cal_acc_cv,3)
            wandb.log({'train_loss':loss/x_train.shape[0],'train_accuracy':acc,'val_loss':val_loss/self.x_cv.shape[0],'val_accuracy':val_acc})
            print('  loss = ',loss/x_train.shape[0],'  accuracy = ',acc,'   validation loss= '
                  ,val_loss/self.x_cv.shape[0],'  validation accuaracy= ',val_acc)
            
    def rmsprop(self,x_train,y_train):
        prev_gradients_w=[]
        temp1=[]
        for i in (self.weights_gradients):
            temp1.append(0*i)
        prev_gradients_w=temp1
        prev_gradients_b=[]
        temp2=[]
        for i in (self.biases_gradients):
            temp2.append(0*i)
        prev_gradients_b=temp2
        eps = 1e-2
        n=self.epochs
        for i in range(n):
            print('Epoch---',i+1,end=" ")
            loss = 0
            val_loss=0
            wg=[]
            for i in (self.weights_gradients):
              wg.append(0*i)
            self.weights_gradients = wg
            bg=[]
            for i in (self.biases_gradients):
              bg.append(0*i)
            self.biases_gradients=bg 
            index = 1
            for x,y in zip(x_train,y_train):
                x = x.ravel()
                val=self.forward_propagation(x,y,self.weights,self.biases)
                loss += val
                self.backward_propagation(x,y,self.weights,self.biases)
                condt=index%self.batch
                if condt == 0 or index == x_train.shape[0]:
                    for j in range(len(self.weights)):
                        t1=(1-self.beta) * np.square(self.weights_gradients[j])
                        v_w = (self.beta * prev_gradients_w[j] +t1)
                        t2=(1-self.beta) * np.square(self.biases_gradients[j])
                        v_b = (self.beta * prev_gradients_b[j] +t2)
                        denom_w=(self.weights_gradients[j] /(np.sqrt(v_w + eps)))
                        self.weights[j] -= self.learning_rate * denom_w
                        denom_b=(self.biases_gradients[j] /(np.sqrt(v_b + eps)))
                        self.biases[j] -= self.learning_rate * denom_b
                        prev_gradients_w[j] = v_w
                        prev_gradients_b[j] = v_b
                    wg=[]
                    for i in (self.weights_gradients):
                      wg.append(0*i)
                    self.weights_gradients=wg
                    bg=[]
                    for i in (self.biases_gradients):
                      bg.append(0*i)
                    self.biases_gradients=bg
                index +=1
            for x,y in zip(self.x_cv,self.y_cv):
               x=x.ravel()
               val=self.forward_propagation(x,y,self.weights,self.biases)
               val_loss+=val

            cal_acc=self.calculate_accuracy(x_train,y_train)
            acc=round(cal_acc,3)
            cal_acc_cv=self.calculate_accuracy(self.x_cv,self.y_cv)
            val_acc=round(cal_acc_cv,3)
            wandb.log({'train_loss':loss/x_train.shape[0],'train_accuracy':acc,'val_loss':val_loss/self.x_cv.shape[0],'val_accuracy':val_acc})
            print('  loss = ',loss/x_train.shape[0],'  accuracy = ',acc,'   validation loss= '
                  ,val_loss/self.x_cv.shape[0],'  validation accuaracy= ',val_acc)


    def adam(self,x_train,y_train):
        m_prev_gradients_w=[]
        temp1=[]
        for i in (self.weights_gradients):
            temp1.append(0*i)
        m_prev_gradients_w=temp1
        m_prev_gradients_b=[]
        temp2=[]
        for i in (self.biases_gradients):
            temp2.append(0*i)
        m_prev_gradients_b=temp2

        v_prev_gradients_w=[]
        temp3=[]
        for i in (self.weights_gradients):
            temp3.append(0*i)
        v_prev_gradients_w=temp3
        v_prev_gradients_b=[]
        temp4=[]
        for i in (self.biases_gradients):
            temp4.append(0*i)
        v_prev_gradients_b=temp4
        iter = 1
        n=self.epochs
        for i in range(n):
            print('Epoch---',i+1,end=" ")
            loss = 0
            val_loss=0
            eps = 1e-2
            wg=[]
            for i in (self.weights_gradients):
              wg.append(0*i)
            self.weights_gradients = wg
            bg=[]
            for i in (self.biases_gradients):
              bg.append(0*i)
            self.biases_gradients=bg 
            index = 1
            for x,y in zip(x_train,y_train):
                x = x.ravel()
                val=self.forward_propagation(x,y,self.weights,self.biases)
                loss +=val 
                self.backward_propagation(x,y,self.weights,self.biases)
                condt=index%self.batch
                if condt == 0 or index == x_train.shape[0]:
                    s=len(self.weights)
                    for j in range(s):
                        p1=(1-self.beta1) * self.weights_gradients[j]
                        m_w = (self.beta1 * m_prev_gradients_w[j]) + p1
                        p2=(1-self.beta1) * self.biases_gradients[j]
                        m_b = (self.beta1 * m_prev_gradients_b[j]) + p2
                        p3=(1-self.beta2) * np.square(self.weights_gradients[j])
                        v_w = (self.beta2 * v_prev_gradients_w[j]) + p3
                        p4=(1-self.beta2) * np.square(self.biases_gradients[j])
                        v_b = (self.beta2 * v_prev_gradients_b[j]) + p4
                        denom1=(1-(self.beta1)**iter)
                        m_hat_w = (m_w)/ denom1
                        m_hat_b = (m_b)/denom1
                        denom2=(1-(self.beta2)**iter)
                        v_hat_w = (v_w)/ denom2
                        v_hat_b = (v_b)/denom2
                        t1=(m_hat_w/(np.sqrt(v_hat_w + eps)))
                        self.weights[j] -= self.learning_rate * t1
                        t2=(m_hat_b/(np.sqrt(v_hat_b + eps)))
                        self.biases[j] -= self.learning_rate * t2
                        v1=m_prev_gradients_w[j]
                        m_prev_gradients_w[j] = m_w
                        m_prev_gradients_b[j] = m_b
                        v2=v_prev_gradients_w[j]
                        v_prev_gradients_w[j] = v_w
                        v_prev_gradients_b[j] = v_b
                        wg=[]
                        for i in (self.weights_gradients):
                           wg.append(0*i)
                        self.weights_gradients = wg
                        bg=[]
                        for i in (self.biases_gradients):
                          bg.append(0*i)
                        self.biases_gradients=bg
                    iter += 1
                index +=1
            for x,y in zip(self.x_cv,self.y_cv):
               x=x.ravel()
               val=self.forward_propagation(x,y,self.weights,self.biases)
               val_loss+=val
            cal_acc=self.calculate_accuracy(x_train,y_train)
            acc=round(cal_acc,3)
            cal_acc_cv=self.calculate_accuracy(self.x_cv,self.y_cv)
            val_acc=round(cal_acc_cv,3)
            wandb.log({'train_loss':loss/x_train.shape[0],'train_accuracy':acc,'val_loss':val_loss/self.x_cv.shape[0],'val_accuracy':val_acc})
            print('  loss = ',loss/x_train.shape[0],'  accuracy = ',acc,'   validation loss= '
                  ,val_loss/self.x_cv.shape[0],'  validation accuaracy= ',val_acc)
        

    def nadam(self,x_train,y_train):
        m_prev_gradients_w=[]
        temp1=[]
        for i in (self.weights_gradients):
            temp1.append(0*i)
        m_prev_gradients_w=temp1
        m_prev_gradients_b=[]
        temp2=[]
        for i in (self.biases_gradients):
            temp2.append(0*i)
        m_prev_gradients_b=temp2

        v_prev_gradients_w=[]
        temp3=[]
        for i in (self.weights_gradients):
            temp3.append(0*i)
        v_prev_gradients_w=temp3
        v_prev_gradients_b=[]
        temp4=[]
        for i in (self.biases_gradients):
            temp4.append(0*i)
        v_prev_gradients_b=temp4
        iter = 1
        n=self.epochs
        for i in range(n):
            print('Epoch---',i+1,end=" ")
            loss = 0
            val_loss=0
            eps = 1e-2
            wg=[]
            for i in (self.weights_gradients):
              wg.append(0*i)
            self.weights_gradients = wg
            bg=[]
            for i in (self.biases_gradients):
              bg.append(0*i)
            self.biases_gradients=bg 
            index = 1
            for x,y in zip(x_train,y_train):
                x = x.ravel()
                val=self.forward_propagation(x,y,self.weights,self.biases)
                loss += val
                self.backward_propagation(x,y,self.weights,self.biases)
                condt=index % self.batch
                if condt == 0 or index == x_train.shape[0]:
                    s=len(self.weights)
                    for j in range(s):
                        p1=(1-self.beta1) * self.weights_gradients[j]
                        m_w = (self.beta1 * m_prev_gradients_w[j]) + p1
                        p2=(1-self.beta1) * self.biases_gradients[j]
                        m_b = (self.beta1 * m_prev_gradients_b[j]) + p2
                        p3=(1-self.beta2) * np.square(self.weights_gradients[j])
                        v_w = (self.beta2 * v_prev_gradients_w[j]) + p3
                        p4=(1-self.beta2) * np.square(self.biases_gradients[j])
                        v_b = (self.beta2 * v_prev_gradients_b[j]) + p4
                        denom1=(1-(self.beta1)**iter)
                        m_hat_w = (m_w)/ denom1
                        m_hat_b = (m_b)/denom1
                        denom2=(1-(self.beta2)**iter)
                        v_hat_w = (v_w)/ denom2
                        v_hat_b = (v_b)/denom2
                        t3=(1-self.beta1) * self.weights_gradients[j]
                        m_dash_w = self.beta1 * m_hat_w + t3
                        t4=(1-self.beta1) * self.biases_gradients[j]
                        m_dash_b = self.beta1 * m_hat_b + t4
                        t1=(m_dash_w/(np.sqrt(v_hat_w + eps)))
                        self.weights[j] -= self.learning_rate * t1
                        t2=(m_dash_b/(np.sqrt(v_hat_b + eps)))
                        self.biases[j] -= self.learning_rate * t2
                        v1=m_prev_gradients_w[j]
                        m_prev_gradients_w[j] = m_w
                        v2=m_prev_gradients_b[j]
                        m_prev_gradients_b[j] = m_b
                        v_prev_gradients_w[j] = v_w
                        v_prev_gradients_b[j] = v_b
                        wg=[]
                        for i in (self.weights_gradients):
                           wg.append(0*i)
                        self.weights_gradients = wg
                        bg=[]
                        for i in (self.biases_gradients):
                          bg.append(0*i)
                        self.biases_gradients=bg
                    iter += 1
                index +=1
            for x,y in zip(self.x_cv,self.y_cv):
               x=x.ravel()
               val=self.forward_propagation(x,y,self.weights,self.biases)
               val_loss+=val
            cal_acc=self.calculate_accuracy(x_train,y_train)
            acc=round(cal_acc,3)
            cal_acc_cv=self.calculate_accuracy(self.x_cv,self.y_cv)
            val_acc=round(cal_acc_cv,3)
            wandb.log({'train_loss':loss/x_train.shape[0],'train_accuracy':acc,'val_loss':val_loss/self.x_cv.shape[0],'val_accuracy':val_acc})
            print('  loss = ',loss/x_train.shape[0],'  accuracy = ',acc,'   validation loss= '
                  ,val_loss/self.x_cv.shape[0],'  validation accuaracy= ',val_acc)
    
    def calculate_accuracy(self,X,Y,flag=False):
        count = 0
        for i in range(len(X)):
            if self.predict(X[i]) == Y[i]:
                count+=1
            if flag:
              self.conf_matrix[self.predict(X[i])][Y[i]]+=1
        if flag:
          wandb.log({'Confusion matrix': wandb.plots.HeatMap(self.actual_labels, self.predicted_labels, self.conf_matrix, show_text=True)})
        return count/len(X)

    def predict(self,x):
        n=len(self.layers)
        x = x.ravel()
        self.activations[0] = x
        for i in range(n-2):
            if self.activation_func == "sigmoid":
                val=self.sigmoid(np.matmul(self.weights[i].T,self.activations[i])+self.biases[i])
                self.activations[i+1] = val
            elif self.activation_func == "tanh":
                val=self.tanh(np.matmul(self.weights[i].T,self.activations[i])+self.biases[i])
                self.activations[i+1] = val
            elif self.activation_func == "relu":
                val=self.relu(np.matmul(self.weights[i].T,self.activations[i])+self.biases[i])
                self.activations[i+1] = val

        self.activations[n-1] = self.softmax(np.matmul(self.weights[n-2].T,self.activations[n-2])+self.biases[n-2])

        return np.argmax(self.activations[-1])

wandb.login()
wandb.init(project = args.wandb_project,entity = args.wandb_entity)

nn = Neural_network(x_train,y_train,784,hidden_layers_size=args.hidden_size,hidden_layers=args.num_layers,output_dim=10,learning_rate=args.learning_rate,batch_size=args.batch_size,
                    epochs=args.epochs,activation_func=args.activation,optimizer=args.optimizer,weight_init=args.weight_init,decay_rate=args.weight_decay,loss=args.loss)



