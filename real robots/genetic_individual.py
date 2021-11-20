import random
import numpy
from action_network import ActionNetwork
from prediction_network import PredictionNetwork

class GeneticIndividual:
    """ represents an individual in a genetic population

    The individual has both an action and a prediction network.

    Usage:
    Call *action* with the current sensor values in order to get the calculated action values.
    Then call *predict* with the current sensor values (the same as for *action*) in order to get the predicted sensor values for the next time step.
    In the next time step you may first call *storeSensor*. Now you can call *evaluate* in order to get the score in [0,1) for this individual.

    Please note that the *predict* function will automatically input the last given action combined with the current sensor values to the prediction network.
    Therefore it is crucial to call *action* before calling *predict*!
    Also make sure that before calling *evaluate* you have called *action*, *predict* and *storeSensor* the same amount of times.

    """

    def __init__(self, amountSensors, amountActions, amountHiddenAction, amountHiddenPrediction, activationFunctionAction, activationFunctionPrediction):
        """ creates a new individual with random weights

        An individual has an action network and a prediction network.
        The amount of input values for the action and prediction networks is the sum of both the *amountSensors* and the *amountActions*.
        The amount of output values of the action network is the amount of actions.
        The amount of output values of the prediction values equals *amountSensors*.

        Arguments:
        amountSensors -- the amount of sensors this individual has
        amountActions -- the amount of actions this individual should predict
        amountHiddenAction -- amount of hidden nodes in the action network
        amountHiddenPrediction -- amount of hidden nodes in the prediction network
        activationFunctionAction -- activation function used in the action network
        activationFunctionPrediction -- activation function used in the prediction network

        """

        self.amountSensors = amountSensors
        self.amountActions = amountActions
        self.amountHiddenAction = amountHiddenAction
        self.amountHiddenPrediction = amountHiddenPrediction

        self.activationFunctionAction = activationFunctionAction
        self.activationFunctionPrediction = activationFunctionPrediction

        self.actionNetwork = ActionNetwork((amountSensors + amountActions), amountHiddenAction, amountActions, activationFunctionAction)
        self.predictionNetwork = PredictionNetwork((amountSensors + amountActions), amountHiddenPrediction, amountSensors, activationFunctionPrediction)

        self.reset()

    def action(self, input):
        """ inputs given vector to the action network and returns network's output """
        vector = numpy.concatenate((input, self.givenAction), 0)  
        self.givenAction = self.actionNetwork.input(vector)
        return self.givenAction

    def predict(self, input):
        """ inputs given vector and last given action to the prediction network and returns network's output """
        vector = numpy.concatenate((input, self.givenAction), 0)  
        self.givenPredictions.append(self.predictionNetwork.input(vector))

        return self.givenPredictions[-1]

    def storeSensor(self, sensor):
        """ stores the actual sensor values for currently given prediction """
        self.actualSensor.append(sensor)

    def reset(self):
        """ resets stored given predictions and actual sensor data """
        self.givenAction = numpy.ones((self.amountActions, 1))
        self.actualSensor = []
        self.givenPredictions = []

    def evaluate(self):
        """ returns fitness value of this individual according to predicted and actual sensor data """
        #diffVectorArray = [1 - numpy.square((self.givenPredictions[i] - self.actualSensor[i])) for i in range(len(self.actualSensor))]
        # 1 - mean absolute error 
        diffVectorArray = [1 - numpy.absolute((self.givenPredictions[i] - self.actualSensor[i])) for i in range(len(self.actualSensor))]
        diffArray = [numpy.sum(x) for x in diffVectorArray]
        diff = sum(diffArray)

        return diff / (len(self.actualSensor) * self.amountSensors)

    def mutate(self, rate):
        """ mutates a copy of this individual and returns it

        Arguments:
        rate -- a real number in [0,1) specifiying the probability for each number in genome to be mutated; should not be much greater than 0.3

        """
        genomeAction = self.actionNetwork.toGenome()
        genomePrediction = self.predictionNetwork.toGenome()

        # mutate
        genomeAction = [x if random.random()>=rate else x + random.uniform(-0.4, 0.4) for x in genomeAction]
        genomePrediction = [x if random.random()>=rate else x + random.uniform(-0.4, 0.4) for x in genomePrediction]

        # recombine
        copy = GeneticIndividual(self.amountSensors, self.amountActions, self.amountHiddenAction, self.amountHiddenPrediction, self.activationFunctionAction, self.activationFunctionPrediction)
        copy.actionNetwork.fromGenome(numpy.array(genomeAction))
        copy.predictionNetwork.fromGenome(numpy.array(genomePrediction))

        return copy
