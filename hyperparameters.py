"""
Number of epochs determines the number of times the entire training dataset is passed through the network.
Adjusted based on network complexity and observed training performance.
"""
num_epochs = 25

"""
Initial learning rate sets the step size at the start of training. 
Adjusted based on network complexity and observed training performance.
"""
inital_learning_rate = 0.025

"""
Decay rate for the learning rate, determining how quickly the learning rate decreases during training. 
A higher decay rate reduces the learning rate more rapidly. 
Adjusted this parameter to control the convergence speed of the training process.
"""
decay_rate = 2.2

"""
Input dimension specifies the size of the input features. 
Matches the number of features in your dataset. 
"""
input_dim = 385

"""
Dimensions of hidden layers in the network, representing the capacity of the network to learn complex representations. 
Choose sizes based on the complexity of the task and the amount of data. 
Balance between sufficient capacity for learning and avoiding overfitting.
"""
hidden_dim = 256
hidden_dim1 = 128
hidden_dim2 = 64
hidden_dim3 = 32

"""
The number of neurons in the output layer of the network. 
For binary classification, a single output neuron (often with a sigmoid activation) is typical.
"""
output_dim = 1
