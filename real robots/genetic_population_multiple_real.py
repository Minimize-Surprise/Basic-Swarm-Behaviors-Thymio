import struct
import numpy
import random
import parameters
from genetic_individual import GeneticIndividual
#from controller import Supervisor
from state import State, write_csv
import time

counter = 0

class GeneticPopulation:
    """ represents a population in terms of 1+1 evolution

    Usage:
    First create a new instance of this class.

    If this instance is running on a client:
        After that you only need to call *execute_client* with the current sensor values. The returned numpy vector is the calculated action to be performed.

    If this instance is running on the master:
        After that you only need to call *execute_master* at every time step.

    """

    def __init__(self, controller, maxAge, postEvalTime, reEval, evals, amountSensors, amountActions, amountHiddenAction, amountHiddenPrediction, mutateRate, activationFunctionAction, activationFunctionPrediction, reEvalWeight):
        """ initializes a population with a king and a mutant

        The population consists of a king (the best seen mutant so far) and a currently being tested mutant. Both are instances of the *GeneticIndividual* class.
        The mutant will be fed all sensor values and is responsible for all actions and predictions.

        Please note, that all arguments except *controller*, *maxAge* and *maxEra* are directly forwarded to the constructor of the genetic individuals.

        Arguments:
        controller -- controller controlling the robot, used to determine whether the instance is a client or a master and for sending and receiving messages
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

        self.MASTER_WAIT_PUFFER = 20 # puffer for waiting for delayed messages (in timesteps)

        self.maxAge = maxAge
        self.postEvalTime = postEvalTime
        self.time = -1
        self.reEval = reEval 
        self.evals = evals
        self.evalNo = -1
        self.mutateRate = mutateRate
        
        # first king and first mutant 
        self.king = GeneticIndividual(amountSensors, amountActions, amountHiddenAction, amountHiddenPrediction, activationFunctionAction, activationFunctionPrediction)
        self.mutant = self.king.mutate(self.mutateRate)
        
        self.scoreKing = 0
        self.scoreTemp = None 
        self.filename = "results/genomes"
        self.POST_EVAL = 0 
        self.reevalWeightNew = reEvalWeight
        self.initdelay = 50
        self.tmpGenome = ""
        self.countRetry = 0
        self.startTime = -1
        
        self.controller = controller
        self.state = State.INIT
        self.init = True
        self.evalCount = 0 # how many genomes were evaluated so far

        if self.controller.getMaster() == True:
            #self.state = State.INIT
            self.evaluationScores = []
 
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

    def _evaluate_master(self, fail):
        """ evaluates the mutant tested on all clients and maybe kills the king """
        
        if fail:
            print("fail")
            if parameters.enableDataTracking:
                self.evaluation_listener(-2,-2)
            self.evaluationScores = []
            
        else: 
            # calculate total score
            self.scoreMutant = sum(self.evaluationScores) / len(self.evaluationScores)
        
            # calculate score for re-evaluation case 
            if self.scoreTemp is not None:
                self.scoreMutant = self.reevalWeightNew * self.scoreMutant + (1.0 - self.reevalWeightNew) * self.scoreTemp 

            #print("score of mutant: " + str(self.scoreMutant) + " | score of old king: " + str(self.scoreKing))
            # log scores 
            if self.evaluation_listener and parameters.enableDataTracking:
                self.evaluation_listener(self.scoreKing, self.scoreMutant)
        
            # mutant replaces current king 
            if self.scoreMutant >= self.scoreKing:
                self.king = self.mutant
                self.scoreKing = self.scoreMutant
                # store/print genome
                if parameters.enableDataTracking:
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
            print("Post Evaluation")

        if self.evalCount != self.evals+1:   
            self._distribute(self.mutant, fail)
        else:
            if self.state != State.STOP:
                print("STOP")
            self.state = State.STOP
        
    def _distribute(self, individual, fail):
        """ sends a genetic individual encoded via genome to clients """

        genomeAction = individual.actionNetwork.toGenome()
        genomePrediction = individual.predictionNetwork.toGenome()

        listAction = [str(x) for x in genomeAction]
        listPrediction = [str(x) for x in genomePrediction]
        
        stamp = time.time() + 1

        # increase quantity of evaluated genomes
        if not fail:
            self.evalCount += 1
            print(self.evalCount)
            
        msg = str(self.evalCount) + "####" + str(stamp) + "####" + str(self.POST_EVAL) + "####" + "@".join(listAction) + "####" + "@".join(listPrediction)
        self.controller.emit(msg)

        self.state = State.WAIT
        
    def _evaluate_client(self, lastSensor):
        """ evaluates the genome and sends it to the master """

        self.mutant.storeSensor(lastSensor)
        self.scoreMutant = self.mutant.evaluate()
        msg = str(self.evalNo) + "####" + str(self.scoreMutant)
        print(msg)
        self.controller.emit(msg) #self.scoreMutant)# double
        self.countRetry = 0

        self.state = State.WAIT

    def _receive_genome(self, msg):
        """ called when the client received a genome and restarts the client using this genome """
        length = len(msg)
        received = self.tmpGenome + msg[0]  # if we already received part of the genome last step add the rest
        for i in range(length-1):
            received = received + msg[i+1]
            
        try:
            [evalNo, stamp, flag, listAction, listPrediction] = received.split("####")  # check if genome is complete
        except:
            self.state = State.WAIT  # try again next timestep
            print("receive split fail")
            self.tmpGenome = received  # save what we already received
            return False

        # update time for post-evaluation
        try:
            self.evalNo = int(evalNo)
        except:
            self.state = State.WAIT
            print("receive fail")
            self.tmpGenome = received
            return False
        
        # update start time for evaluation
        try:
            self.startTime = float(stamp)
        except:
            self.state = State.WAIT
            print("receive fail")
            self.tmpGenome = received
            return False

        # update time for post-evaluation
        try: 
            self.POST_EVAL = int(flag)
        except:
            self.state = State.WAIT
            print("receive fail")
            self.tmpGenome = received
            return False
            
        if self.POST_EVAL:
            self.maxAge = self.postEvalTime
            
        genomeAction = listAction.split("@")
        try:
            genomeAction = [float(x) for x in genomeAction]  # super rare edge case this doen't work with incomplete genome
        except:
            self.state = State.WAIT
            print("receive fail")
            self.tmpGenome = received
            return False
        genomePrediction = listPrediction.split("@")
        try:
            genomePrediction = [float(x) for x in genomePrediction]  # super rare edge case this doen't work with incomplete genome
        except:
            self.state = State.WAIT
            print("receive fail")
            self.tmpGenome = received
            return False
        
        if (len(genomeAction) != parameters.genLengthAction or len(genomePrediction) != parameters.genLengthPred):
            #check if genome is complete
            self.state = State.WAIT  
            print("receive fail")
            self.tmpGenome = received
            return False
            
        self.mutant.actionNetwork.fromGenome(numpy.array(genomeAction))
        self.mutant.predictionNetwork.fromGenome(numpy.array(genomePrediction))
        print(int(evalNo))
        self.mutant.reset()
        if self.tmpGenome != "":
            print("all good now")
        self.tmpGenome = ""
        self.time = -1
        #self.evalCount += 1
        self.state = State.RUN

    def execute_master(self):
        """ evaluates the mutant when max age is reached """
        global counter
        
        # wait until all clients are connected then wait 5 more seconds
        if self.state == State.INIT:
            msg = self.controller.receive()
            if counter >= parameters.ROBOTS:
                self.initdelay = self.initdelay - 1
                print(self.initdelay)
            elif msg:
                counter += 1
                print(msg)
            
            if self.initdelay == 0:
                msg = "True"
                self.controller.emit(msg) #sent START FLAG
                
            if self.initdelay < 0:
                self.state = State.WAIT
                self._distribute(self.mutant,False)
                
            return True
        
        if self.state == State.WAIT:

            msg = self.controller.receive()

            if msg:
                # received first score of a mutant
                received = msg
                length = len(received)
                for i in range(length):
                    print(received[i])
                    msg = received[i].split("####")
                    if int(msg[0]) == self.evalCount:
                        self.evaluationScores.append(float(msg[1]))

                if len(self.evaluationScores) == parameters.ROBOTS:  #check if every bot sent a score
                    self.time = 0
                    self._evaluate_master(False) # evaluate the mutant
                        
            else:
                self.time += 1
                if self.time > self.maxAge + 50:  # if there is no word from a thymio
                    self.time = 0
                    print("no scores received")
                    self._evaluate_master(True) # send flag that not all thymios got the last genome so send same genome again in next eval step

    def execute_client(self, sensor):
        """ calculates an action and evaluates the mutant when max age is reached """
        if self.state == State.INIT:
            msg = self.controller.receive()
            
            if not msg:
                return None, None
            else:
                length = len(msg)
                for i in range(length):
                    print(msg[i])
                
                self.state = State.WAIT
                
                return None, None
                
        elif self.state == State.WAIT:
            msg = self.controller.receive()
            #self.countRetry += 1

            if not msg:
                self.tmpGenome = ""
            else:
                flag = self._receive_genome(msg)
                
            return None, None # abort and keep waiting
        
        elif self.state == State.RUN and time.time() < self.startTime: #self.countRetry <= 15:  # make sure to wait 5 timesteps so all clients start simultaniously
            #self.countRetry += 1
            return None, None
                
        self.time = self.time + 1

        if self.time >= self.maxAge:
            self._evaluate_client(sensor)
            self.time = -1  
            return None, None # abort and start waiting 

        action,pred = self._feed(sensor)
        return action, pred
