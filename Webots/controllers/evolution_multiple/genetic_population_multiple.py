import struct
import numpy
import random
from genetic_individual import GeneticIndividual
from controller import Supervisor
from state import State, write_csv  


class GeneticPopulation:
    """ represents a population in terms of 1+1 evolution

    Usage:
    First create a new instance of this class.

    If this instance is running on a slave:
        After that you only need to call *execute_slave* with the current sensor values. The returned numpy vector is the calculated action to be performed.

    If this instance is running on the master:
        After that you only need to call *execute_master* at every time step.

    """

    def __init__(self, controller, maxAge, postEvalTime, reEval, evals, amountSensors, amountActions, amountHiddenAction, amountHiddenPrediction, mutateRate, activationFunctionAction, activationFunctionPrediction, reEvalWeight):
        """ initializes a population with a king and a mutant

        The population consists of a king (the best seen mutant so far) and a currently being tested mutant. Both are instances of the *GeneticIndividual* class.
        The mutant will be fed all sensor values and is responsible for all actions and predictions.

        Please note, that all arguments except *controller*, *maxAge* and *maxEra* are directly forwarded to the constructor of the genetic individuals.

        Arguments:
        controller -- controller controlling the robot, used to determine whether the instance is a slave or a master and for sending and receiving messages
        maxAge -- evaluation length of genome/mutant 
        postEvalTime -- length of postevaluation 
        reEval -- percentage chance of re-evaluation of current king 
        evals -- maximum number of evaluations / cycles / generations 
        amountSensors -- amount of sensor values (needed for the genetical individuals)
        amountHiddenAction -- the number of nodes in the hidden layer of the action network (needed for the genetical individuals)
        amountHiddenPrediction --  the number of nodes in the hidden layer of the prediction network /nneded for the genetical (needed for the genetical individuals)
        mutateRate -- the rate at which mutation takes place, denoted by a real number in the interval [0,1) but usually not much greater than 0.3 (needed for the genetical individuals)
        activationFunctionAction -- activation function used in the action networks (needed for the genetical individuals)
        activationFunctionPrediction -- activation function used in the prediction networks (needed for the genetical individuals)
        reEvalWeight -- weight of newly determined score after re-evaluating king; old score weighted with (1-reEvalWeight) 
        """

        self.MASTER_WAIT_PUFFER = 10 # puffer for waiting for delayed messages (in timesteps)

        self.maxAge = maxAge
        self.postEvalTime = postEvalTime
        self.time = -1
        self.reEval = reEval 
        self.evals = evals 
        self.mutateRate = mutateRate
        
        # first king and first mutant 
        self.king = GeneticIndividual(amountSensors, amountActions, amountHiddenAction, amountHiddenPrediction, activationFunctionAction, activationFunctionPrediction)
        self.mutant = self.king.mutate(self.mutateRate)
        
        self.scoreKing = 0
        self.scoreTemp = None 
        self.filename = "results/genomes" 
        self.POST_EVAL = 0 
        self.reevalWeightNew = reEvalWeight 
        
        self.controller = controller
        self.state = State.WAIT

        if self.controller.master:
            self.evaluationScores = []
            self.evalCount = 0 # how many genomes were evaluated so far 
            self._distribute(self.mutant)
            self.controller.robot.simulationSetMode(Supervisor.SIMULATION_MODE_FAST) 
 
    def set_evaluation_listener(self, listener):
        """
        sets the function to be called when evaluation happens

        function will get two parameters: first is the fitness of the king, second the one of the mutant
        set listener function to undefined in order to disable

        """

        self.evaluation_listener = listener

    def _feed(self, sensor):
        """ feeds the mutant with fresh sensor values and returns the calculated action """        

        if self.time > 0: # shift in sensor value storage to compare matching sensors and predictions 
            self.mutant.storeSensor(sensor)

        action = self.mutant.action(sensor)
        pred = self.mutant.predict(sensor)

        return action, pred

    def _evaluate_master(self):
        """ evaluates the mutant tested on all slaves and maybe kills the king """
        
        # calculate total score 
        self.scoreMutant = sum(self.evaluationScores) / len(self.evaluationScores)
        
        # calculate score for re-evaluation case 
        if self.scoreTemp is not None:
            self.scoreMutant = self.reevalWeightNew * self.scoreMutant + (1.0 - self.reevalWeightNew) * self.scoreTemp 

        #print("score of mutant: " + str(self.scoreMutant) + " | score of old king: " + str(self.scoreKing))
        # log scores 
        if self.evaluation_listener:
            self.evaluation_listener(self.scoreKing, self.scoreMutant)
        
        # mutant replaces current king 
        if self.scoreMutant >= self.scoreKing:
            self.king = self.mutant
            self.scoreKing = self.scoreMutant
            # store/print genome
            line = ",".join(str(x) for x in self.king.actionNetwork.toGenome())
            write_csv(self.filename, line)
            line = ",".join(str(x) for x in self.king.predictionNetwork.toGenome()) + str("\n")
            write_csv(self.filename, line) 
             
        self.evaluationScores = []
        
        # re-evalate with a chance of reEval percent 
        if random.random() < self.reEval: 
            self.mutant = self.king
            self.mutant.reset()             
            self.scoreTemp = self.scoreKing             
            self.scoreKing = -1 
        
        else:
            self.mutant = self.king.mutate(self.mutateRate)
            self.scoreTemp = None  
            
        # POST EVALUATION  
        if self.evalCount == self.evals:
            self.mutant = self.king
            self.mutant.reset()     
            self.scoreKing = -1 
            self.scoreTemp = None # don't calculate mixed score 
            self.maxAge = self.postEvalTime # extend period for re-eval / video 
            self.POST_EVAL = 1 
            
            # reposition robots to new initial positions 
            self.controller.reposition_robots()
            
            # START VIDEO
            movie = 'results/run.mp4'
            self.controller.robot.simulationSetMode(Supervisor.SIMULATION_MODE_REAL_TIME)
            self.controller.robot.movieStartRecording(movie, 800, 600, 0, 100, 1, False) # start recording of video 
        
        if self.evalCount != self.evals+1:   
            self._distribute(self.mutant) 
        else:           
            self.controller.robot.movieStopRecording() # stop everything 
            self.state = State.STOP                             
        
    def _distribute(self, individual):
        """ sends a genetic individual encoded via genome to slaves """

        genomeAction = individual.actionNetwork.toGenome()
        genomePrediction = individual.predictionNetwork.toGenome()

        listAction = [str(x) for x in genomeAction]
        listPrediction = [str(x) for x in genomePrediction]
        
        msg = str(self.POST_EVAL) + "####" + "@".join(listAction) + "####" + "@".join(listPrediction)
        self.controller.emit(struct.pack("10000s", msg.encode("utf-8")))
        
        # increase quantity of evaluated genomes 
        self.evalCount += 1 

        #print("sent genome")
        self.state = State.WAIT
        

    def _evaluate_slave(self, lastSensor):
        """ evaluates the genome and sends it to the master """

        self.mutant.storeSensor(lastSensor)
        self.scoreMutant = self.mutant.evaluate()

        self.controller.emit(struct.pack("d",self.scoreMutant)) # double 

        self.state = State.WAIT

    def _receive_genome(self, msg):
        """ called when the slave received a genome and restarts the slave using this genome """

        #print("received genome")
        received = struct.unpack("10000s", msg)[0].decode("utf-8").rstrip("\x00")
        # update post eval flag and genome 
        [flag, listAction, listPrediction] = received.split("####")
        
        # update time for post-evaluation 
        self.POST_EVAL = int(flag)
        if self.POST_EVAL:
            self.maxAge = self.postEvalTime
            
        genomeAction = listAction.split("@")
        genomeAction = [float(x) for x in genomeAction]
        genomePrediction = listPrediction.split("@")
        genomePrediction = [float(x) for x in genomePrediction]

        self.mutant.actionNetwork.fromGenome(numpy.array(genomeAction))
        self.mutant.predictionNetwork.fromGenome(numpy.array(genomePrediction))
        self.mutant.reset()

        self.time = -1
        self.state = State.RUN

    def execute_master(self):
        """ evaluates the mutant when max age is reached """

        if self.state == State.WAIT or self.state == State.WAIT_PUFFER:

            msg = self.controller.receive()

            if msg is not None:
                # received first score of a mutant

                received = struct.unpack("d",msg)[0]
                self.evaluationScores.append(received)
                self.execute_master() # collect all scores received in this timestep 
                self.time = 0
                self.state = State.WAIT_PUFFER # wait for delayed scores

            elif self.state == State.WAIT_PUFFER:

                self.time = self.time + 1

                if self.time > self.MASTER_WAIT_PUFFER:
                    self._evaluate_master() # evaluate the mutant
                

    def execute_slave(self, sensor):
        """ calculates an action and evaluates the mutant when max age is reached """
        
        if self.state == State.WAIT:

            msg = self.controller.receive()

            if msg is not None:
                self._receive_genome(msg)
            else:
                return None, None # abort and keep waiting

        self.time = self.time + 1

        if self.time >= self.maxAge:
            self._evaluate_slave(sensor)
            self.time = -1  
            return None, None # abort and start waiting 

        action,pred = self._feed(sensor)
        return action, pred
