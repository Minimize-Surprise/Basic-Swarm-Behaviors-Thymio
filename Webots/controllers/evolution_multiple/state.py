from enum import Enum

class State(Enum):
    """ enum for the implicit state machine used in *GeneticPopulation* """
    RUN = 1 # indicates that the individual can calculate action values normally
    WAIT = 2 # wait for command or message
    WAIT_PUFFER = 3 # wait for some delayed message
    STOP = 4 # stop run 


def write_csv(filename, line):
    """ writes the given line to specified csv file

    Arguments:
    filename -- the file the line should be written to; the file ending '.csv' will be added within the method, so you should really omit '.csv' in filename ;)
    line -- the line to be written

    """
    file = open(filename + ".csv", "a")
    file.write(line + "\n")
    file.close()