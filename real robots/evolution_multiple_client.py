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
from clientME import Client

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
        """ initialises values for behavior and logfiles

        """

        # create a genetic population
        self.population = GeneticPopulation(self, parameters.EVAL_TIME, parameters.POST_EVAL_TIME, parameters.RE_EVAL_PROB, parameters.EVALS, parameters.SENSORS, parameters.ACTIONS, parameters.HIDDEN_ACTION, parameters.HIDDEN_PRED, parameters.MUT_RATE, tanh, sigmoid, parameters.RE_EVAL_WEIGHT)
      
        
        if parameters.enableDataTracking:
            # init log file for predictions and sensors
            self.filename = "results/pred"
             
            if not os.path.isfile(self.filename):
                write_csv(self.filename, "obstacle avoidance,pred0 (t+1),pred1 (t+1),pred2 (t+1),pred3 (t+1),pred4 (t+1),pred5 (t+1),pred6 (t+1),predg0 (t+1),predg1 (t+1),s0 (t),s1 (t),s2 (t),s3 (t),s4 (t),s5 (t),s6 (t),sg0 (t),sg1 (t),m0 selected,m1 selected,m0 real,m1 real")
    
    def getMaster(self):
        return False
        
    def _log(self, line):
        """ writes the line to the logfile """
        write_csv(self.filename, line)

    def emit(self, msg):
        """ sends a message """
        client.sendMessage(msg)

    def receive(self):
        # read buffer of incomming messages
        tmpBuffer = client.readBuffer()
        return(tmpBuffer)

    def control(self):
        """ calculates motor values and sets them using 1+1 evolution distributed across a master and his clients """

        # transform sensor values into a numpy vector
        sensors = robot.getAllSensors()
                        
        action, pred = self.population.execute_client(sensors)  # this is the line containing the 1+1 evolution magic
        obstacle_avoidance = 0 
        
        if action is not None:
            motor = [action[0][0] * parameters.MAX_SPEED, action[1][0] * parameters.MAX_SPEED] # retrieve the calculated action values
            tmp = motor.copy() # copy motor values for logging purposes 
            
            # check if we are likely to hit a wall
            avoidResult = self._hwp(motor, sensors)  
            
            if avoidResult is not None:
                motor = avoidResult # if we are likely to hit an obstacle, we may want to prevent this 
                obstacle_avoidance = 1 
            else:
                pass  # well, yeah, you may just delete this else case
                
            if self.population.POST_EVAL and action is not None: # log values during post-evaluation
                if parameters.enableDataTracking:
                    write_csv(self.filename, str(obstacle_avoidance)+","+str(pred[0][-1])+","+str(pred[1][-1])+","+str(pred[2][-1])+","+str(pred[3][-1]) +","+str(pred[4][-1])+","+str(pred[5][-1])+","+str(pred[6][-1])+","+str(pred[7][-1])+","+str(pred[8][-1]) +","+str(sensors[0][-1])+","+str(sensors[1][-1])+","+str(sensors[2][-1])+","+str(sensors[3][-1])+","+str(sensors[4][-1])+","+str(sensors[5][-1])+","+str(sensors[6][-1])+","+str(sensors[7][-1])+","+str(sensors[8][-1])+","+str(tmp[0])+","+str(tmp[1])+","+str(motor[0])+","+str(motor[1]))
        
        else: # no action if eval time is over 
            motor = [0,0]

        # set motor values
        robot.setMotorValues(motor[0], motor[1])

        return True


    def _hwp(self, motor, sensors):
        """ stop robot if it wants to drive into other robot or wall """ 
        
        # robot stopped - no HWP necessary 
        if motor[0] == 0 and motor[1] == 0:
            return None 
           
        # straight drive or large turning radius large (i.e. robot would drive relatively straight) 
        if (motor[0] == motor[1]) or np.abs((motor[0]+motor[1])/(motor[1]-motor[0])) >= 1.0:
        
            # driving forward - check for horizontal obstacle in front or wall
            if (motor[0] > 0 and motor[1] > 0) and (sensors[0][0] > 0.75 or sensors[1][0] > 0.75 or sensors[2][0] > 0.75 or sensors[3][0] > 0.75 or sensors[4][0] > 0.75 or sensors[7][0] >= 0.55 or sensors[8][0] >= 0.55):
                return [0,0]   
            
            # driving backwards - check for horizontal obstacle in back or wall
            if (motor[0] < 0 and motor[1] < 0) and (sensors[5][0] > 0.75 or sensors[6][0] > 0.75 or sensors[7][0] >= 0.55 or sensors[8][0] >=  0.55):
                return [0,0] 
                
        # no HW protection necessary 
        return None 



if __name__ == '__main__':
    # Get reference to the robot.
    robot = Robot()
    
    # needed for communication with other robots
    client = Client()

    contr = Controller()
    # GObject -> calls controller every 100 ms
    loop = GObject.MainLoop()
    handle = GObject.timeout_add(100, contr.control)
    
    # exit via control c
    try:
        loop.run()
    except KeyboardInterrupt:
        robot.setMotorValues(0, 0)
        client.closeClient()
        print("exit programm")
    finally:
        robot.setMotorValues(0, 0)
        client.closeClient()
