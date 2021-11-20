import unittest
import numpy
from genetic_individual import *

class TestGeneticIndividual(unittest.TestCase):

    def test_evaluate(self):

        individual = GeneticIndividual(1, 1, 1, 1, lambda x : x, lambda x : x)

        sensor1 = numpy.array([[1]])
        action1 = individual.action(sensor1)
        prediction1 = individual.predict(sensor1)

        sensor2 = numpy.array([[2]])
        individual.storeSensor(sensor2)
        action2 = individual.action(sensor2)
        prediction2 = individual.predict(sensor2)

        sensor3 = numpy.array([[3]])
        individual.storeSensor(sensor3)

        score = individual.evaluate()
        correctScore = ((1 - abs(prediction1[0][0] - sensor2[0][0])) + (1 - abs(prediction2[0][0] - sensor3[0][0]))) / (2*1)
        self.assertEqual(score, correctScore, "evaluate does not evaluate correctly!")

    def test_mutate(self):
        individual = GeneticIndividual(1, 1, 1, 1, lambda x : x, lambda x : x)
        mutant = individual.mutate(0)
        self.assertTrue(numpy.array_equal(individual.actionNetwork.toGenome(), mutant.actionNetwork.toGenome()), "rate=0 should produce a copy")
        self.assertTrue(numpy.array_equal(individual.predictionNetwork.toGenome(), mutant.predictionNetwork.toGenome()), "rate=0 should produce a copy")

        mutant = individual.mutate(1)
        self.assertFalse(numpy.array_equal(individual.actionNetwork.toGenome(), mutant.actionNetwork.toGenome()), "rate=0 should produce a copy")
        self.assertFalse(numpy.array_equal(individual.predictionNetwork.toGenome(), mutant.predictionNetwork.toGenome()), "rate=0 should produce a copy")


if __name__ == '__main__':
    unittest.main()
