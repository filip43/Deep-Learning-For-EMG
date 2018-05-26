
# # Implementing a deep neural network using tensorflow
# In this notebok i will implement a 3 hidden layer neural network and feed in all the neccesary data. We will be focused on using S1_A1_E1.mat which means we are only concerned with one subject.

# ## 1. Import all the neccesary packages
import math
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import scipy
import scipy.io as sio
import time
import py3nvml
py3nvml.grab_gpus(num_gpus=1, gpu_select=[1])

# ## 2. Load the data from .mat file and split it into a training (70%) and test (30%) data.
def load_dataset():
    #Load the emg data
    XData = sio.loadmat('emgShuffled.mat')
    X_orig = (np.array((XData['emgShuffled']))).T
    X_train_orig = X_orig[:,0:int(0.7*X_orig.shape[1])]
    X_test_orig = X_orig[:,int(0.7*X_orig.shape[1])+1::]
    #Flatten the EMG data
    X_train_orig = X_train_orig/np.amax(X_train_orig)
    X_test_orig = X_test_orig/np.amax(X_test_orig)
    #Load the labels
    YData = sio.loadmat('yShuffled.mat')
    Y_orig = (np.array((YData['yShuffled']))).T
    Y_train_orig = Y_orig[:,0:int(0.7*Y_orig.shape[1])]
    Y_test_orig = Y_orig[:,int(0.7*Y_orig.shape[1])+1::]
    return X_train_orig, Y_train_orig, X_test_orig, Y_test_orig

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = load_dataset()

# ## 4. Couple of functions that are needed for the implementation, and do not need much explaining

def random_mini_batches(X, Y, mini_batch_size):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m/mini_batch_size)) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def relu(Z):
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    """
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)

    return A


def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [10, 1])
    
    z3 = forward_propagation(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction

def forward_propagation_with_numpy(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = np.dot(W1, X) + b1                      # Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)                                    # A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2                   # Z2 = np.dot(W2, a1) + b2
    A2 = relu(Z2)                                    # A2 = relu(Z2)
    Z3 = np.dot(W3,A2) + b3                     # Z3 = np.dot(W3,A2) + b3
    
    return Z3


# ## 5. Convert all the labels to one-hot vector and flatten the EMG values AND SAY HOW MANY CLASSES SUPER IMPORTANT
no_of_classes = 13
Y_train = convert_to_one_hot(Y_train_orig, no_of_classes)
Y_test = convert_to_one_hot(Y_test_orig, no_of_classes)
# Normalize EMG vectors
X_train = X_train_orig
X_test = X_test_orig

print ("Y_train shape: " + str(Y_train.shape))
print ("Y_test shape: " + str(Y_test.shape))


# ## 6. Create Placeholders
# Create placeholders for `X` and `Y`. This will allow you to later pass your training data in when you run your session.
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    """

    X = tf.placeholder(tf.float32, shape = [n_x, None])
    Y = tf.placeholder(tf.float32,shape = [n_y, None])
    keep_prob = tf.placeholder(tf.float32)
    
    return X, Y, keep_prob

# ## 7. Initialise the paremeters

def initialize_parameters(sizeOfInput, layer1Size, layer2Size, no_of_classes):
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [layer1Size, sizeOfInput]
                        b1 : [layer1Size, 1]
                        W2 : [layer2Size, layer1Size]
                        b2 : [layer2Size, 1]
                        W3 : [no_of_classes, layer2Size]
                        b3 : [no_of_classes, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """
            
    W1 = tf.get_variable("W1", [layer1Size,sizeOfInput], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [layer1Size,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [layer2Size, layer1Size], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [layer2Size, 1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [no_of_classes, layer2Size], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [no_of_classes, 1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters


# ## 8. Forward propagation in tensorflow 
# 
# Now we will implement the forward propagation module in tensorflow. The function will take in a dictionary of parameters and it will complete the forward pass. The functions you will be using are: 
# 
# - `tf.add(...,...)` to do an addition
# - `tf.matmul(...,...)` to do a matrix multiplication
# - `tf.nn.relu(...)` to apply the ReLU activation

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    
    Z1 = tf.add(tf.matmul(W1,X), b1)                                             # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                              # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1), b2)                                              # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2), b3)                                              # Z3 = np.dot(W3,Z2) + b3
    
    return Z3

# forward_propagation_with_dropout

def forward_propagation_with_dropout(X, parameters, keep_prob):
    """
    Implements the forward propagation: LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.
    
    Arguments:
    X -- input dataset, of shape (2, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
                    W1 -- weight matrix of shape (20, 2)
                    b1 -- bias vector of shape (20, 1)
                    W2 -- weight matrix of shape (3, 20)
                    b2 -- bias vector of shape (3, 1)
                    W3 -- weight matrix of shape (1, 3)
                    b3 -- bias vector of shape (1, 1)
    keep_prob - probability of keeping a neuron active during drop-out, scalar
    
    Returns:
    A3 -- last activation value, output of the forward propagation, of shape (1,1)
    cache -- tuple, information stored for computing the backward propagation
    """
    
    # retrieve parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]
    
    Z1 = tf.add(tf.matmul(W1,X), b1)                                             # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                             # A1 = relu(Z1)
    drop_out_1 = tf.nn.dropout(A1, keep_prob)
    Z2 = tf.add(tf.matmul(W2,drop_out_1), b2)                                              # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                              # A2 = relu(Z2)
    drop_out_2 = tf.nn.dropout(A2, keep_prob)
    Z3 = tf.add(tf.matmul(W3,drop_out_2), b3)                                              # Z3 = np.dot(W3,Z2) + b3
    
    return Z3

# ## 9. Compute cost
# 
# it is very easy to compute the cost using:
# ```python
# tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = ..., labels = ...))
# ```


def compute_cost(Z3, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = labels))
    
    return cost

# ## 10. Backward propagation & parameter updates
# Create an "`optimizer`" object. Call this object along with the cost when running the tf.session. When called, it will perform an optimization on the given cost with the chosen method and learning rate.
# For instance, for gradient descent the optimizer would be:
# ```python
# optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
# ```
# To make the optimization you would do:
# ```python
# _ , c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
# ```
# 
# This computes the backpropagation by passing through the tensorflow graph in the reverse order. From cost to inputs.
# # Build the model
def model(X_train, Y_train, X_test, Y_test, learning_rate,
          num_epochs, minibatch_size, print_cost, layer1Size, layer2Size, keep_probability):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y, keep_prob = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(n_x, layer1Size, layer2Size, no_of_classes)
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation_with_dropout(X, parameters, keep_prob = keep_probability)
   
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)
   
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    #Initialise the seeion with right parameters
    conf = tf.ConfigProto(allow_soft_placement = True)#log_device_placement = True
    conf.gpu_options.allow_growth = True
    # Start the session to compute the tensorflow graph
    with tf.Session(config=conf) as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        """
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
	"""
        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters

start = time.time()
parameters = model(X_train, Y_train, X_test, Y_test, learning_rate = 0.001,
          num_epochs = 1000, minibatch_size = 256, print_cost = True, layer1Size=36, layer2Size=32, keep_probability = 0.65)
end =time.time()

print("The time taken to train parameters is ", end-start)

# # Obtain the accuracies per each class

# ## Training Set
Z3_train = forward_propagation_with_numpy(X_train, parameters)

a = tf.placeholder(tf.float32,shape = [X_train.shape[1],])
b = tf.placeholder(tf.float32,shape = [X_train.shape[1],])

c = tf.confusion_matrix(labels = a, predictions = b)

with tf.Session() as sess:
    confusion_matrix = c.eval(feed_dict={a:sess.run(tf.argmax(Y_train)),b:sess.run(tf.argmax(Z3_train))})
    print("The confusion matrix for training data is:\n")
    print(np.true_divide(confusion_matrix.diagonal(), np.sum(confusion_matrix, axis = 0)))


# ## Test Set

Z3_test = forward_propagation_with_numpy(X_test, parameters)

a = tf.placeholder(tf.float32,shape = [X_test.shape[1],])
b = tf.placeholder(tf.float32,shape = [X_test.shape[1],])

c = tf.confusion_matrix(labels = a, predictions = b)

with tf.Session() as sess:
    confusion_matrix = c.eval(feed_dict={a:sess.run(tf.argmax(Y_test)),b:sess.run(tf.argmax(Z3_test))})
    print(confusion_matrix)
    print("The confusion matrix for test data is:\n")    
    print(np.true_divide(confusion_matrix.diagonal(), np.sum(confusion_matrix, axis = 0)))


