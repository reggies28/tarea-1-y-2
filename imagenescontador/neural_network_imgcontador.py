# python notebook for Make Your Own Neural Network
# code for a 3-layer neural network, and code for learning the MNIST dataset
# (c) Tariq Rashid, 2016
# license is GPLv2

import numpy
# scipy.special for the sigmoid function expit()
import scipy.special
# library for plotting arrays
import matplotlib.pyplot
# ensure the plots are inside this notebook, not an external window
import copy

# neural network class definition
class neuralNetwork:
    
    # initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = numpy.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        # learning rate
        self.lr = learningrate
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        
        pass
    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target - actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors) 
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass
    
    # query the neural network
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

### MNIST ###

print("TRAINING IMAGES")

# number of input, hidden and output nodes
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.1

# create instance of neural network
nmnist = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

# load the mnist test data CSV file into a list
test_data_file = open("img_contador/mnist_test_10.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

# test the neural network

def test():
    # scorecard for how well the network performs, initially empty
    scorecard = []
    # arrays para almacenar los labels y las salidas de la red    
    label_arr = numpy.zeros(shape=(0,10))
    output_arr = numpy.zeros(shape=(0,10))    
    print()

    # go through all the records in the test data set
    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct answer is first value
        correct_label = int(all_values[0])
        list_label = [0 for i in range(output_nodes)]
        list_label[correct_label] = 1
        list_label = numpy.asfarray(list_label).reshape(1, 10)
        label_arr = numpy.append(label_arr, list_label, axis=0)
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # query the network
        outputs = nmnist.query(inputs)
        output_arr = numpy.append(output_arr, outputs.reshape(1, 10), axis=0)
        # the index of the highest value corresponds to the label        
        label = numpy.argmax(outputs)
        # append correct or incorrect to list
        if (label == correct_label):
            # network's answer matches correct answer, add 1 to scorecard
            scorecard.append(1)
        else:
            # network's answer doesn't match correct answer, add 0 to scorecard
            scorecard.append(0)
            pass
        
        pass

    # calculate the performance score, the fraction of correct answers
    scorecard_array = numpy.asarray(scorecard)
    return str(scorecard_array.sum() / scorecard_array.size * 100) + "%"


# load the mnist training data CSV file into a list
training_data_file = open("img_contador/mnist_train_100.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network

# epochs is the number of times the training data set is used for training
epochs = 30

for e in range(epochs):
    print("\nIteración", e+1)
    # go through all records in the training data set
    for record in training_data_list:        
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        nmnist.train(inputs, targets)
        pass    
    print("Accuracy:", test())    
    pass

### CONTADOR ###

# number of input, hidden and output nodes
input_nodes = 20
hidden_nodes = 300
output_nodes = 20

# learning rate
learning_rate = 0.08

# create instance of neural network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# load the mnist training data CSV file into a list
training_data_file = open("img_contador/train_contador.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# train the neural network
print("TRAINING\n")

# epochs is the number of times the training data set is used for training
epochs = 500

for e in range(epochs):
    # imprimir el numero de iteracion
    print("Iteracion", e + 1)    
    # go through all records in the training data set    
    for record in training_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs        
        inputs0 = all_values[2] + all_values[3]
        inputs = [0] * 20
        inputs[int(inputs0[0])] = 1
        inputs[int(inputs0[1]) + 10] = 1
        inputs = numpy.asfarray(inputs)
        # create the target output values
        targets0 = all_values[0] + all_values[1]
        targets = [0] * 20
        targets[int(targets0[0])] = 1
        targets[int(targets0[1]) + 10] = 1
        n.train(inputs, targets) 
        pass
    pass

## UNION REDES

print("\nTESTING IMAG + NEXT\n")

def testRedes():
    test_data_file = open("img_contador/test_redes.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    for record in test_data_list:
        # split the record by the ',' commas
        all_values = record.split(',')
        # correct value
        correct = all_values[0]
        # first digit
        input1 = all_values[1:785]
        input1 = numpy.asfarray(input1)
        first = nmnist.query(input1).flatten()
        maxf = numpy.argmax(first)
        first = [0] * 10
        first[maxf] = 1        
        # second digit
        input2 = all_values[785:1569]
        input2 = numpy.asfarray(input2)
        second = nmnist.query(input2)
        maxs = numpy.argmax(second)
        second = [0] * 10
        second[maxs] = 1
        # create the target output values
        number = []
        number.append(first)
        number.append(second)
        number = numpy.asfarray(number)
        number = number.flatten()
        res = n.query(number).flatten()

        res1 = res[:10].tolist()
        res2 = res[10:].tolist()
        print("Resultado final:", res1.index(max(res1)), res2.index(max(res2)))
        pass
    pass
    
testRedes()






    
    
