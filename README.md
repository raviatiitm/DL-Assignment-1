# DL-Assignments
`CS6910` IIT Madras Deep Learning Assignment

1. Question 1
    1. Just by running the code (in google colab), we can see the output of the code in wandb.

2. Question 2
    1. 1. Here, a class called Neural network was built with the following parameters: x train, y train, input dimension, hidden layer size, hidden layer output dimension, batch size, activation function, learning rate, decay rate, beta, beta1, beta2, optimizer, and weight init
    2. we should pass the training data (x train, y train), input dimension size, hidden layer size, number of hidden layers, output dimension, batch size, number of epochs, which activation function we want to use, learning rate, decay rate, beta, beat1, beta2 for the Adam optimizer, which optimizer we want to use, and which weight initializer we want to use when creating an object to this class.
    3. The gradient descent optimizer, activation functions for sigmoid, tanh, and relu, from which we can choose any one, as well as random and Xavier weight initializers, are all implemented in this python file.
    4.While generating an object of this class, we can use any batch size, any learning rate, and any epochs.

3. Question 3
    1. We use optimizers such as sgd (stocastic gradient descent), momentum, Nestrov Accelerated Gradient, RMS Prop, ADAM, and NADAM in this case.
    2.  If we provide optimizer="rmsprop" when creating an object for this class, the RMSProp optimizer will be applied for doing calculations.
    3. Similarly 
      * If we want to use SGD optimizer then we use optimizer="sgd"
      * If we want to use Momentum optimizer then we use optimizer="momentum"
      * If we want to use NAG optimizer then we use optimizer="nestrov"
      * If we want to use Adam optimizer then we use optimizer="adam"
      * If we want to use NAdam optimizer then we use optimizer="nadam"
    
 4. Question 4
    1. To start, we divided the train data into nine equal halves. 10% of the data is used for cross validation, while 90% of the data is used for training.
    2. After that, we configured several options in sweep config to set the wandb sweep function.
    3. We can start seeing the output in our wandb project by running the code below.
    ```
    wandb.agent(sweep_id,train)
    ```
 
 5. Question 7
    1. Test data were utilised to plot the confusion matrix.
    2. To plot the confusion matrix, we selected the model that provided the best validation accuracy.
    3. The model is as follows:
      - optimizer : nadam
	  - activation : relu
	  - batch size : 64
	  - hidden layers : 2 
	  - learning rate : 0.002
	  - weight decay : 0.1
    5. By running the below code we can see plot the confusion matrix in our wandb project:
    ```
       wandb.init(project='DL_Assignment1',entity='cs22m069')
       nn = Neural_network(x_test,y_test,784,64,2,10,learning_rate=0.002,batch_size = 64,epochs=10,
                    activation_func="relu",optimizer="nadam",weight_init="xavier",decay_rate=0.1)
       nn.calculate_accuracy(x_test,y_test,True)
    ```

6. Question 8
    1. Just as we did with cross entropy loss in the prior questions, we introduced mean squared error loss in this file.
    2. Using 2 configurations, we now contrasted this squared error loss with cross entropy loss.
    3. Mean squared error loss often does not perform better than cross entropy loss in terms of cross validation accuracy, while there are rare instances when cross         validation  accuracy will be quite close.


7. Question 10
	We used three of the best congruences from the fashion mnist dataset's previous congruences to import Mnist data from the Keras dataset into this file.
8. Train.py
	1.You must use the api key to log into your wandb account from the terminal in order to use train.py.
	2.Please enter proper credentials like project name and entity name in the terminal's parameters.
	3.By including a member function in the class NeuralNetwork and adding the initializer's name to the Initialization list dictionary, you can add new initialization methods.
	4.By including a member function in the class NeuralNetwork and entering the function's name into the dictionary for the activation list, you can add additional activation functions.
	5.By inserting new code in the area designated by the fit technique, new optimization methods can be added.
	6.Please use the following example to call train.py with the proper argument:
		python train.py -e 1 -w_i Xavier -b 32 -o nag -nhl 3 -sz 128 -d fashion_mnist -l mean_squared_error -lr 0.001


