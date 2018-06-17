from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from werkzeug.exceptions import abort
from numpy import exp, array, random, dot

class NeuralNetwork():
  def __init__(self):
      # seed the random number generator
    random.seed(1)

    # model a single neuron with 3 input connections and 1 output connection.
    # assign random weights to a 3 x 1 matrix, with values in range of -1 to 1, and mean 0
    self.synaptic_weights = 2 * random.random((3, 1)) - 1

  # The S shaped curve function
  # We pass the weighted sum of the inputs through this function to normalize it between 0 and 1
  def __sigmoid(self, x):
    return 1 / (1 + exp(-x))


  # The derivative of the Sigmoid
  # Gradient of the Sigmoid
  # Indicates how confident we are about the existing weight
  def __sigmoid_derivative(self, x):
    return x * (1 - x)

  # We train the NN through a process of trial and error, by adjusting the synaptic weights each time
  def train(self, inputs, outputs, iterations):
    for iteration in iterations:
        # Pass the training set through our NN (a single neuron)
      output = self.think(inputs)

      # Calculate the error (difference between desired and predicted output)
      error = outputs - output

      # Multiply the error by input and gain the gradient Sigmoid curve
      # Less confident wieghts are adjusted more
      # Inputs, which are zeros, do not cause changes to the weights
      adjustments = dot(inputs.T, error * self.__sigmoid_derivative(output))

      # Adjust the weights
      self.synaptic_weights += adjustments

  # The NN thinks
  def think(self, inputs):
    return self.__sigmoid(dot(inputs, self.synaptic_weights))

simple_nn = Blueprint('simple_nn', __name__)

nn = NeuralNetwork()
starting_synapse = nn.synaptic_weights
training_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
training_outputs =  array([[0, 1, 1, 0]]).T # transposed
nn.train(training_inputs, training_outputs, range(10000))

@simple_nn.route('/simple_nn')
def index():
    data = {
        'starting_synapse': starting_synapse,
        'inputs': training_inputs,
        'trained_outputs': training_outputs
    }
    return render_template('simple_nn/index.html', data=data)

@simple_nn.route('/simple_nn/<string:new_input>')
def think(new_input):
    new_input = array(list(new_input), dtype=int)
    guess = nn.think(new_input)

    data = {
        'starting_synapse': starting_synapse,
        'inputs': training_inputs,
        'trained_outputs': training_outputs,
        'new_input': new_input,
        'guess': guess ,
        'rounded_guess': round(guess[0])
    }
    return render_template('simple_nn/think.html', data=data)
