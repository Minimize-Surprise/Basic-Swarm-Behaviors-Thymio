"""  
Minimize Surprise - Evolution - Thymio

If the robot controlled by this controller should be the master robot, the name must contain 'master'.
Consequently, all other robots may not contain 'master' in their names.
"""

import sys
import random
import os.path
from controller import Robot, Emitter, Receiver, Supervisor 
import numpy as np
from genetic_population_multiple import GeneticPopulation
from state import State, write_csv 


# Constants for the Thymio's speed and sensors
MAX_SPEED = 6 #9.53 - we don't go full speed 
MAX_HORIZONTAL_SENSOR = 4500.0 	# maximum horizontal sensor value
MAX_GROUND_SENSOR = 1100.0   # maximum ground sensor value
ROBOTS = 10 # number of robots 

# EXPERIMENT PARAMETERS 
EVALS = 1000 
POSTEVAL = False 
EVAL_TIME = 1000 
POST_EVAL_TIME = 10000
RE_EVAL_PROB = 0.2 
RE_EVAL_WEIGHT = 0.2 
SENSORS = 9 # 5 front horizontal + 2 back horizontal + 2 ground
ACTIONS = 2 # left and right wheel 
HIDDEN_ACTION = 7 #5 
HIDDEN_PRED = 10 #6
MUT_RATE = 0.1 
ARENA_X = None
ARENA_Y = None 
ROBR = 0.082
ROBOTS = 10 

def createRandom(): 
    """
    create random positions and headings
    constant set per generation 
    different values per robot and repetition 
    """
    
    global rot 
    global pos 
    
    # reset 
    pos = [] 
    rot = [] 
        
    # headings 
    rot = [2*np.pi*random.random() for i in range(ROBOTS)]
    
    # position             
    for i in range(ROBOTS):
        free = False 
    
        while not free: 
            x = random.random() * (ARENA_X - 2* ROBR) - 0.5 * (ARENA_X - 2* ROBR)
            y = random.random() * (ARENA_Y - 2* ROBR) - 0.5 * (ARENA_Y - 2* ROBR)
            free = True 
        
            for k in range(0, len(pos)):                                        
                dist = np.linalg.norm([pos[k][0] - x, pos[k][2] - y ])
                                        
                if dist < 2*ROBR: 
                    free = False 
       
        pos.append([x, 0, y]) 


def sigmoid(x):
    """ returns the value of the sigmoid function evaluated at all elements of x """
    return 1 / (1 + np.exp(-x))
    
    
def tanh(x):
    """ returns the value of the sigmoid function evaluated at all elements of x """
    return np.tanh(x)


class Controller():
    """
    Controller for thymio simulation using 1+1 evolution  distributed across a master and his slaves

    If the robot controlled by this controller should be the master robot, the robot's name must contain 'master'.
    Consequently, all other robots may not contain 'master' in their names.

    Usage:
    Simply call *control* at every time step. Well, after having created an object of course.

    """

    def __init__(self, robot, name, master, emitter, receiver):
        """ initialises values for avoid behavior and logfiles

        Arguments:
        robot -- reference to supervisor or robot instance 
        name -- robot name 
        master -- true iff this instance is the master robot
        emitter -- reference to webots emitter
        receiver -- reference to webots receiver

        """
        global ARENA_X 
        global ARENA_Y 
        
        self.robot = robot 
        # name for file I/O 
        self.name = name 

        # store reference, because they are very useful ;)
        self.master = master
        self.emitter = emitter
        self.receiver = receiver

        # create a genetic population
        self.population = GeneticPopulation(self, EVAL_TIME, POST_EVAL_TIME, RE_EVAL_PROB, EVALS, SENSORS, ACTIONS, HIDDEN_ACTION, HIDDEN_PRED, MUT_RATE, tanh, sigmoid, RE_EVAL_WEIGHT)

        if self.master:
            # get floor size 
            floor = self.robot.getFromDef('Floor')
            floor_size = floor.getField('size').getSFVec2f()
            ARENA_X = floor_size[0]
            ARENA_Y = floor_size[1]
        
            # init log files
            self.filename = "results/run"

            write_csv("results/parameters", "evaluation length," + str(self.population.maxAge)) 
            write_csv("results/parameters", "post-evaluation length," + str(POST_EVAL_TIME)) 
            write_csv("results/parameters", "re-evaluation probability," + str(self.population.reEval)) 
            write_csv("results/parameters", "re-evaluation weight (new score)," + str(self.population.reevalWeightNew)) 
            write_csv("results/parameters", "mutation rate," + str(self.population.mutateRate)) 
            write_csv("results/parameters", "sensors," + str(self.population.king.amountSensors)) 
            write_csv("results/parameters", "actions," + str(self.population.king.amountActions))
            write_csv("results/parameters", "hidden nodes action ANN," + str(self.population.king.amountHiddenAction))
            write_csv("results/parameters", "hidden nodes prediction ANN," + str(self.population.king.amountHiddenPrediction))
            write_csv("results/parameters", "transfer function action ANN,tanh")
            write_csv("results/parameters", "transfer function prediction ANN,sigmoid")         
            write_csv("results/parameters", "robots," + str(ROBOTS))         
            write_csv("results/parameters", "arena size x," + str(ARENA_X))         
            write_csv("results/parameters", "arena size y," + str(ARENA_Y))         
            

            if not os.path.isfile("results/trajectory.csv"):
                write_csv("results/trajectory", "robot,translation0,translation1,translation2,rotation0,rotation1,rotation2,rotation3")

            self._log("king,mutant")

            # register evaluation listener for logging
            self.population.set_evaluation_listener(lambda x,y: self._log(str(x) + "," + str(y)))
                        
            self.reposition_robots() 
        
        else: # slave
             # init log file for predictions and sensors 
             self.filename = "results/pred_" + str(self.name) 
             
             if not os.path.isfile(self.filename):
                 write_csv(self.filename, "obstacle avoidance,pred0 (t+1),pred1 (t+1),pred2 (t+1),pred3 (t+1),pred4 (t+1),pred5 (t+1),pred6 (t+1),predg0 (t+1),predg1 (t+1),s0 (t),s1 (t),s2 (t),s3 (t),s4 (t),s5 (t),s6 (t),sg0 (t),sg1 (t),m0 selected,m1 selected,m0 real,m1 real")
             

    def reposition_robots(self): 
        createRandom()
        for i in range(1, ROBOTS+1):   
            slave = self.robot.getFromDef('T' + str(i))
            slave.getField('translation').setSFVec3f(pos[i-1])
            slave.getField('rotation').setSFRotation([0, 1, 0, rot[i-1]]) 
                 
    def _log(self, line):
        """ writes the line to the logfile """
        write_csv(self.filename, line)

    def emit(self, msg):
        """ sends a message """
        self.emitter.send(msg)

    def receive(self):
        """ receive a (one!) message and returns it, None if no message received """
        if self.receiver.getQueueLength() > 0:
            msg = self.receiver.getData()
            self.receiver.nextPacket()
            return msg
        return None

    def control(self):
        """ manage master and slave logic

        on a slave: calculates motor values and sets them using 1+1 evolution encapsulated in a genetic population
        on the master:  waits for evaluation scores and distributs new genomes

        """        
        if self.master:
            self.control_master()
        else:
            self.control_slave()

    def control_master(self):
        """ wait for evaluation scores and distribute new genome """        
        if self.population.POST_EVAL:
            if self.robot.movieIsReady(): # Quit simulation when run is over 
                self.robot.simulationQuit(1)
                
            if self.population.state != State.STOP: 
                # log trajectory for each robot 
                for i in range(1, ROBOTS+1): 
                    slave = self.robot.getFromDef('T' + str(i))
                    translationValues = slave.getField('translation').getSFVec3f()
                    rotationValues = slave.getField('rotation').getSFRotation() 
                    write_csv("results/trajectory", 'T' + str(i) + ',' + str(translationValues[0]) + ',' + str(translationValues[1]) + ',' + str(translationValues[2]) + ',' + str(rotationValues[0]) + ',' + str(rotationValues[1]) + ',' + str(rotationValues[2]) + ',' + str(rotationValues[3]))
        
        # receive fitness values, determine overall fitness, distribute new genomes           
        self.population.execute_master()
        
        # don't move        
        motor = [0,0]
        leftMotor.setVelocity(motor[0])
        rightMotor.setVelocity(motor[1])
        
        return True

    def control_slave(self):
        """ calculates motor values and sets them using 1+1 evolution distributed across a master and his slaves """

        # transform sensor values into a numpy vector
        sensors = np.array([
                            [outerLeftSensor.getValue()/MAX_HORIZONTAL_SENSOR],
                            [centralLeftSensor.getValue()/MAX_HORIZONTAL_SENSOR],
                            [centralSensor.getValue()/MAX_HORIZONTAL_SENSOR],
                            [centralRightSensor.getValue()/MAX_HORIZONTAL_SENSOR],
                            [outerRightSensor.getValue()/MAX_HORIZONTAL_SENSOR],
                            [backLeftSensor.getValue()/MAX_HORIZONTAL_SENSOR],
                            [backRightSensor.getValue()/MAX_HORIZONTAL_SENSOR], 
                            [groundLeftSensor.getValue()/MAX_GROUND_SENSOR],
                            [groundRightSensor.getValue()/MAX_GROUND_SENSOR]
                        ])
                        
        action, pred = self.population.execute_slave(sensors)  # this is the line containing the 1+1 evolution magic
        obstacle_avoidance = 0 
        
        if action is not None:
            motor = [action[0][0] * MAX_SPEED, action[1][0] * MAX_SPEED] # retrieve the calculated action values
            tmp = motor.copy() # copy motor values for logging purposes 
            
            # check if we are likely to hit a wall
            avoidResult = self._hwp(motor, sensors)  
            
            if avoidResult is not None:
                motor = avoidResult # if we are likely to hit an obstacle, we may want to prevent this 
                obstacle_avoidance = 1 
            else:
                pass  # well, yeah, you may just delete this else case
                
            if self.population.POST_EVAL: # log values during post-evaluation                                    
                write_csv(self.filename, str(obstacle_avoidance)+","+str(pred[0][-1])+","+str(pred[1][-1])+","+str(pred[2][-1])+","+str(pred[3][-1])
                                         +","+str(pred[4][-1])+","+str(pred[5][-1])+","+str(pred[6][-1])+","+str(pred[7][-1])+","+str(pred[8][-1])
                                         +","+str(sensors[0][-1])+","+str(sensors[1][-1])+","+str(sensors[2][-1])+","+str(sensors[3][-1])+","
                                         +str(sensors[4][-1])+","+str(sensors[5][-1])+","+str(sensors[6][-1])+","+str(sensors[7][-1])+","+str(sensors[8][-1])
                                         +","+str(tmp[0])+","+str(tmp[1])+","+str(motor[0])+","+str(motor[1]))  
        
        else: # no action if eval time is over 
            motor = [0,0]

        # set motor values
        leftMotor.setVelocity(motor[0])
        rightMotor.setVelocity(motor[1])

        return True


    def _hwp(self, motor, sensors):
        """ stop robot if it wants to drive into other robot or wall """ 
        
        # robot stopped - no HWP necessary 
        if motor[0] == 0 and motor[1] == 0:
            return None 
           
        # straight drive or large turning radius large (i.e. robot would drive relatively straight) 
        if (motor[0] == motor[1]) or np.abs((motor[0]+motor[1])/(motor[1]-motor[0])) >= 1.0:
        
            if (motor[0] > 0 and motor[1] > 0) and (sensors[0][0] > 0.9 or sensors[1][0] > 0.9 or sensors[2][0] > 0.9 or sensors[3][0] > 0.9 or sensors[4][0] > 0.9 or sensors[7][0] < 0.01 or sensors[8][0] < 0.01): 
                return [0,0]   
            
            # driving backwords - check for horizontal obstacle in back or wall 
            if (motor[0] < 0 and motor[1] < 0) and (sensors[5][0] > 0.9 or sensors[6][0] > 0.9 or sensors[7][0] < 0.01 or sensors[8][0] < 0.01): 
                return [0,0] 
                
        # no HW protection necessary 
        return None 



if __name__ == '__main__':
    # Get reference to the robot.
    if str(sys.argv[1]) == "slave":
        robot = Robot()
    else:
        robot = Supervisor() 

    # Get simulation step length
    timeStep = int(robot.getBasicTimeStep())

    # Get left and right wheel motors
    leftMotor = robot.getMotor("motor.left")
    rightMotor = robot.getMotor("motor.right")

    # Get distance sensors
    outerLeftSensor = robot.getDistanceSensor("prox.horizontal.0")
    centralLeftSensor = robot.getDistanceSensor("prox.horizontal.1")
    centralSensor = robot.getDistanceSensor("prox.horizontal.2")
    centralRightSensor = robot.getDistanceSensor("prox.horizontal.3")
    outerRightSensor = robot.getDistanceSensor("prox.horizontal.4")
    
    backLeftSensor = robot.getDistanceSensor("prox.horizontal.5")
    backRightSensor = robot.getDistanceSensor("prox.horizontal.6")
    
    groundLeftSensor = robot.getDistanceSensor("prox.ground.0")
    groundRightSensor = robot.getDistanceSensor("prox.ground.1")

    # Enable distance sensors
    outerLeftSensor.enable(timeStep)
    centralLeftSensor.enable(timeStep)
    centralSensor.enable(timeStep)
    centralRightSensor.enable(timeStep)
    outerRightSensor.enable(timeStep)
    
    backLeftSensor.enable(timeStep)
    backRightSensor.enable(timeStep)
    
    groundLeftSensor.enable(timeStep)
    groundRightSensor.enable(timeStep)

    # Disable motor PID control mode
    leftMotor.setPosition(float('inf'))
    rightMotor.setPosition(float('inf'))

    # check if this controller is run on the master robot
    print("master" in robot.getName()) 
    controller = Controller(robot, robot.getName(), "master" in robot.getName(), robot.getEmitter("emitter"), robot.getReceiver("receiver"))
    controller.receiver.enable(timeStep)

    # beginning of execution
    while(robot.step(timeStep) != -1):
        controller.control()
