import numpy

class ActionNetwork:
    """ represents a feedforward network with one hidden layer

    Usage:
    Call the constructor with desired amount of nodes. The *amountIn* value may be equal to the amount of sensors.
    Then call *input* with a vector of those sensor values in order to retrieve the output vector.

    """

    def __init__(self, amountIn, amountHidden, amountOut, activationFunction):
        """ initialize a new ActionNetwork with specified sizes and random matrices

        this class represents a simple feed forward network

        Arguments:
        amountIn -- amount of input values for neural network
        amountHidden -- amount of hidden nodes in the only hidden layer
        amountOut -- amount of output values of the neural network
        activationFunction -- reference to a function applied to the sum of the incoming values for each node

        Usage:
        call *input(vector)* to input a vector to the network and get the network output

        """
        self.amountIn = amountIn + 1 
        self.amountHidden = amountHidden  
        self.amountOut = amountOut
        self.activationFunction = activationFunction
        self._constructRandomMatrices()

    def _constructRandomMatrices(self):
        """ generates random matrices with values in [-1,1) suspect to change via fromGenom """
        self.hidden = 2 * numpy.random.rand(self.amountHidden, self.amountIn) - 1
        self.out = 2 * numpy.random.rand(self.amountOut, (self.amountHidden+1)) - 1 # bias neuron for hidden layer 

    def toGenome(self):
        """ returns the genome representing this network """
        return numpy.concatenate((self.hidden.flatten(), self.out.flatten()))

    def fromGenome(self, genome):
        """ overwrites current matrices with those in given genome

        Arguments:
        genome -- genome to reconstruct matrices from, must match exactly the format corresponding to the dimensions of matrices created in constructor

        """

        # get either genome
        genomeHidden = genome[:self.amountHidden*self.amountIn]
        genomeOut = genome[self.amountHidden*self.amountIn:]
        
        # construct matrices
        self.hidden = genomeHidden.reshape(self.amountHidden, self.amountIn)
        self.out = genomeOut.reshape(self.amountOut, (self.amountHidden+1))

    def input(self, vector):
        """ input a vector to this network and returns network's output

        Arguments:
        vector -- numpy vector with exactly *amountIn* elements

        """
        # add bias to input vector
        vector = numpy.concatenate((vector, [[-1]]), 0)  
        
        # calculate hidden layer outputs 
        tmp = self.activationFunction(self.hidden.dot(vector))
        # add bias for calculation of outputs 
        tmp = numpy.concatenate((tmp, [[-1]]), 0) 

        return self.activationFunction(self.out.dot(tmp))
