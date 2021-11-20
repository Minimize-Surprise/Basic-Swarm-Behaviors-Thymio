import unittest
import numpy
from action_network import *

class TestActionNetwork(unittest.TestCase):

    def test_network(self):
        network = ActionNetwork(2, 3, 1, lambda x : x)

        testVector = numpy.array([[1], [2]])
        resultSum = numpy.array([[3]])
        resultFirst = numpy.array([[1]])
        resultSecond = numpy.array([[2]])

        network.hidden = numpy.array([[1,0], [0,1], [1,1]])

        network.out = numpy.array([[0, 0, 1]])
        self.assertEqual(network.input(testVector), resultSum)

        network.out = numpy.array([[0, 1, 0]])
        self.assertEqual(network.input(testVector), resultSecond)

        network.out = numpy.array([[1, 0, 0]])
        self.assertEqual(network.input(testVector), resultFirst)

    def test_activation(self):
        network = ActionNetwork(2, 3, 1, lambda x : x^2)

        testVector = numpy.array([[1], [2]])
        resultSum = numpy.array([[(3^2)^2]])
        resultFirst = numpy.array([[(1^2)^2]])
        resultSecond = numpy.array([[(2^2)^2]])

        network.hidden = numpy.array([[1,0], [0,1], [1,1]])

        network.out = numpy.array([[0, 0, 1]])
        self.assertEqual(network.input(testVector), resultSum)

        network.out = numpy.array([[0, 1, 0]])
        self.assertEqual(network.input(testVector), resultSecond)

        network.out = numpy.array([[1, 0, 0]])
        self.assertEqual(network.input(testVector), resultFirst)

    def test_genome(self):
        network = ActionNetwork(2, 3, 1, lambda x : x)
        genome = network.toGenome()
        copy = ActionNetwork(2, 3, 1, lambda x : x)
        copy.fromGenome(genome)
        self.assertTrue(numpy.array_equal(network.hidden, copy.hidden), 'fromGenome should be inverse to toGenome')
        self.assertTrue(numpy.array_equal(network.out, copy.out), 'fromGenome should be inverse to toGenome')

if __name__ == '__main__':
    unittest.main()
