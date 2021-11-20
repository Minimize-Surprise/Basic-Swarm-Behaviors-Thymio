import numpy

class PredictionNetwork:
    """ represents a feedforward, recurrent network with one hidden layer

    Usage:
    Call constructor and initialize network with desired parameters.
    Then call *input* with a vector, for example the action values, and retrieve the prediction vector.

    """

    def __init__(self, amountIn, amountHidden, amountOut, activationFunction):
        """ initialize a new PredictionNetwork with specified sizes and random matrices

        Arguments:
        amountIn -- amount of inputs to the network
        amountHidden -- amount of hidden nodes in the only hidden layer
        amountOut -- amount of output nodes
        activationFunction -- reference to a function applied to the sum of all incoming values in each node, the output of this function is the output of the node

        Usage:
        Call *input* to get the networks output

        """
        self.amountIn = amountIn + 1
        self.amountHidden = amountHidden
        self.amountOut = amountOut
        self.hiddenOutput = None
        self.activationFunction = activationFunction
        self._constructRandomMatrices()

    def _constructRandomMatrices(self):
        """ generates random matrices with values in [-1,1) suspect to change via fromGenom """
        self.hidden = 2 * numpy.random.rand(self.amountHidden, self.amountIn) - 1
        self.hidden = numpy.concatenate((self.hidden, numpy.diag( 2 * numpy.random.rand(self.amountHidden) - 1 )), 1)
        self.hiddenOutput = 2 * numpy.random.rand(self.amountHidden, 1) - 1
        self.out = 2 * numpy.random.rand(self.amountOut, (self.amountHidden+1)) - 1

    def toGenome(self):
        """ returns the genome representing this network """
        return numpy.concatenate((self.hidden[:,:self.amountIn].flatten(),
                                    numpy.diag(self.hidden[:,self.amountIn:]),
                                    self.hiddenOutput.flatten(),
                                    self.out.flatten()))

    def fromGenome(self, genome):
        """ overwrites current matrices with those in given genome """

        # get either genome
        genomeHidden = genome[ : self.amountHidden * self.amountIn]
        genomeHiddenDiag = genome[self.amountHidden * self.amountIn : self.amountHidden * self.amountIn + self.amountHidden]
        genomeHiddenOutput = genome[self.amountHidden * self.amountIn + self.amountHidden : self.amountHidden * self.amountIn + 2 * self.amountHidden]
        genomeOut = genome[self.amountHidden * self.amountIn + 2 * self.amountHidden : ]

        # construct matrices
        self.hidden = genomeHidden.reshape(self.amountHidden, self.amountIn)
        self.hidden = numpy.concatenate((self.hidden, numpy.diag(genomeHiddenDiag)), 1)
        self.hiddenOutput = genomeHiddenOutput.reshape(self.amountHidden, 1)
        self.out = genomeOut.reshape(self.amountOut, (self.amountHidden+1))

    def input(self, vector):
        """ input a vector to this network and returns the output of the network

        Arguments:
        vector -- numpy vector that should be input to this network, amount of elements must match *amountIn*

        """

        vector = numpy.concatenate((vector, [[-1]]), 0) # bias 
        self.hiddenOutput = self.activationFunction(self.hidden.dot(numpy.concatenate((vector, self.hiddenOutput), 0)))
        tmp = numpy.concatenate((self.hiddenOutput, [[-1]]),0) # concatenate bias 

        return self.activationFunction(self.out.dot(tmp))
