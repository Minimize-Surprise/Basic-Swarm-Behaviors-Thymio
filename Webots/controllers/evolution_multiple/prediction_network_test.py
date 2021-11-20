import unittest
import numpy
from prediction_network import *

class TestPredictionNetwork(unittest.TestCase):

    def test_network(self):
        network = PredictionNetwork(2, 3, 1, lambda x : x)

        testVector = numpy.array([[1], [2]])
        resultSum = numpy.array([[3]])
        resultFirst = numpy.array([[1]])
        resultSecond = numpy.array([[2]])

        network.hidden = numpy.array([[1,0, 0,0,0], [0,1, 0,0,0], [1,1, 0,0,0]])

        network.out = numpy.array([[0, 0, 1]])
        self.assertEqual(network.input(testVector), resultSum)

        network.out = numpy.array([[0, 1, 0]])
        self.assertEqual(network.input(testVector), resultSecond)

        network.out = numpy.array([[1, 0, 0]])
        self.assertEqual(network.input(testVector), resultFirst)

    def test_network_activation(self):
        network = PredictionNetwork(2, 3, 1, lambda x : x)

        testVector = numpy.array([[1], [2]])
        resultSum = numpy.array([[(3^2)^2]])
        resultFirst = numpy.array([[(1^2)^2]])
        resultSecond = numpy.array([[(2^2)^2]])

        network.hidden = numpy.array([[1,0, 0,0,0], [0,1, 0,0,0], [1,1, 0,0,0]])

        network.out = numpy.array([[0, 0, 1]])
        self.assertEqual(network.input(testVector), resultSum)

        network.out = numpy.array([[0, 1, 0]])
        self.assertEqual(network.input(testVector), resultSecond)

        network.out = numpy.array([[1, 0, 0]])
        self.assertEqual(network.input(testVector), resultFirst)

    def test_network_with_self_weights(self):
        network = PredictionNetwork(2, 3, 1, lambda x : x)

        nullVector = numpy.array([[0], [0]])
        testVector = numpy.array([[1], [2]])
        resultSum = numpy.array([[3]])
        resultFirst = numpy.array([[1]])
        resultSecond = numpy.array([[2]])

        network.hidden = numpy.array([[1,0, 1,0,0], [0,1, 0,1,0], [1,1, 0,0,1]])
        network.hiddenOutput = numpy.array([[0], [0], [0]])

        network.out = numpy.array([[0, 0, 1]])
        self.assertEqual(network.input(testVector), resultSum)

        # now input null vector to check for output conserving

        network.out = numpy.array([[0, 0, 1]])
        self.assertEqual(network.input(nullVector), resultSum)

        network.out = numpy.array([[0, 1, 0]])
        self.assertEqual(network.input(nullVector), resultSecond)

        network.out = numpy.array([[1, 0, 0]])
        self.assertEqual(network.input(nullVector), resultFirst)

        network.hidden = numpy.array([[1,0, 1,0,0], [0,1, 0,1,0], [1,1, 0,0,0.5]])
        network.out = numpy.array([[0, 0, 1]])
        self.assertEqual(network.input(nullVector), 0.5 * resultSum)
        self.assertEqual(network.input(nullVector), 0.25 * resultSum)
        self.assertEqual(network.input(nullVector), 0.125 * resultSum)

    def test_genome(self):
        network = PredictionNetwork(2, 3, 1, lambda x : x)
        genome = network.toGenome()
        copy = PredictionNetwork(2, 3, 1, lambda x : x)
        copy.fromGenome(genome)
        self.assertTrue(numpy.array_equal(network.hidden, copy.hidden), 'fromGenome should be inverse to toGenome')
        self.assertTrue(numpy.array_equal(network.hiddenOutput, copy.hiddenOutput), 'fromGenome should be inverse to toGenome')
        self.assertTrue(numpy.array_equal(network.out, copy.out), 'fromGenome should be inverse to toGenome')

if __name__ == '__main__':
    unittest.main()
