"""  
Minimize Surprise - Evolution - Thymio
"""

import sys
import random
import math
import os.path
import numpy as np
from genetic_population_multiple_real import GeneticPopulation
from state import State, write_csv 
import parameters

from gi.repository import GObject
# imports for robot
from rob import Robot
# imports for connection
from serverMe import Server

def sigmoid(x):
    """ returns the value of the sigmoid function evaluated at all elements of x """
    return 1 / (1 + np.exp(-x))
    
    
def tanh(x):
    """ returns the value of the sigmoid function evaluated at all elements of x """
    return np.tanh(x)


class Controller():
    """
    Controller for thymio simulation using 1+1 evolution  distributed across a master and his clients

    If the robot controlled by this controller should be the master robot, the robot's name must contain 'master'.
    Consequently, all other robots may not contain 'master' in their names.

    Usage:
    Simply call *control* at every time step. Well, after having created an object of course.

    """

    def __init__(self):
        """ initialises values for avoid behavior and logfiles

        Arguments:
        robot -- reference to supervisor or robot instance 
        name -- robot name 
        master -- true iff this instance is the master robot
        emitter -- reference to webots emitter
        receiver -- reference to webots receiver

        """ 

        # create a genetic population
        self.population = GeneticPopulation(self, parameters.EVAL_TIME, parameters.POST_EVAL_TIME, parameters.RE_EVAL_PROB, parameters.EVALS, parameters.SENSORS, parameters.ACTIONS, parameters.HIDDEN_ACTION, parameters.HIDDEN_PRED, parameters.MUT_RATE, tanh, sigmoid, parameters.RE_EVAL_WEIGHT)

        # init log files
        self.filename = "results/run"
            
        if parameters.enableDataTracking:
            write_csv("results/parameters", "SEP=,")
            write_csv("results/parameters", "arenaSizeX," + str(parameters.ARENA_X))
            write_csv("results/parameters", "arenaSizeY," + str(parameters.ARENA_Y))
            write_csv("results/parameters", "robotAmount," + str(parameters.ROBOTS))
            write_csv("results/parameters", "maxAge," + str(self.population.maxAge))
            write_csv("results/parameters", "reEval," + str(self.population.reEval))
            write_csv("results/parameters", "postEvalTime," + str(parameters.POST_EVAL_TIME))
            write_csv("results/parameters", "amountSensors," + str(self.population.king.amountSensors))
            write_csv("results/parameters", "amountActions," + str(self.population.king.amountActions))
            write_csv("results/parameters", "amountHiddenAction," + str(self.population.king.amountHiddenAction))
            write_csv("results/parameters", "amountHiddenPrediction," + str(self.population.king.amountHiddenPrediction))
            write_csv("results/parameters", "mutateRate," + str(self.population.mutateRate))
            write_csv("results/parameters", "transferFuncAction,tanh")
            write_csv("results/parameters", "transferFuncPred,sigmoid")
            
            self._log("SEP=,")
            self._log("king,mutant")

            # register evaluation listener for logging
            self.population.set_evaluation_listener(lambda x,y: self._log(str(x) + "," + str(y)))
                        
             
    def getMaster(self):
        return True

    def _log(self, line):
        """ writes the line to the logfile """
        write_csv(self.filename, line)

    def emit(self, msg):
        """ sends a message """
        server.sendBroadcast(msg)

    def receive(self):
        # read buffer of incomming messages
        tmpBuffer = server.readBuffer()
        return(tmpBuffer)
        

    def control(self):      
        """ wait for evaluation scores and distribute new genome """ 
        
        # receive fitness values, determine overall fitness, distribute new genome
        self.population.execute_master()
        
        # don't move        
        motor = [0,0]
        robot.setMotorValues(motor[0], motor[1])
        return True
          
        
if __name__ == '__main__':
    # needed for communication with robot...
    robot = Robot()

    # needed for communication with other robots
    server = Server()

    contr = Controller()
    # GObject -> calls controller every 100 ms
    loop = GObject.MainLoop()
    handle = GObject.timeout_add(100, contr.control)
    # exit via control c
    try:
        loop.run()
    except KeyboardInterrupt:
        server.closeServer()
        print("exit programm")
    finally:
        server.closeServer()
