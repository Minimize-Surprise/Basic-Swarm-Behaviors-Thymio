# imports for thymio
import dbus
import dbus.mainloop.glib
import sys
from optparse import OptionParser
import parameters
import numpy as np

class Robot:

    def __init__(self):
        self.proxSensorsVal = [0, 0, 0, 0, 0, 0, 0]
        self.groundSensorsVal = [0, 0]
        self.parser = OptionParser()
        self.parser.add_option("-s", "--system", action="store_true", dest="system", default=False,
                          help="use the system bus instead of the session bus")

        (options, args) = self.parser.parse_args()

        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)

        if options.system:
            self.bus = dbus.SystemBus()
        else:
            self.bus = dbus.SessionBus()

        # Create Aseba network
        self.network = dbus.Interface(self.bus.get_object('ch.epfl.mobots.Aseba', '/'),
                                      dbus_interface='ch.epfl.mobots.AsebaNetwork')

    def getAllProxSensors(self):
        self.network.GetVariable("thymio-II", "prox.horizontal", reply_handler=self.get_variables_reply,
                                    error_handler=self.get_variables_error)
        
        return [(self.proxSensorsVal[0] / parameters.MAX_HORIZONTAL_SENSOR),
                (self.proxSensorsVal[1] / parameters.MAX_HORIZONTAL_SENSOR),
                (self.proxSensorsVal[2] / parameters.MAX_HORIZONTAL_SENSOR),
                (self.proxSensorsVal[3] / parameters.MAX_HORIZONTAL_SENSOR),
                (self.proxSensorsVal[4] / parameters.MAX_HORIZONTAL_SENSOR),
                (self.proxSensorsVal[5] / parameters.MAX_HORIZONTAL_SENSOR),
                (self.proxSensorsVal[6] / parameters.MAX_HORIZONTAL_SENSOR)]
    
    def getAllSensors(self):
        # update sensors
        self.network.GetVariable("thymio-II", "prox.horizontal", reply_handler=self.get_variables_reply,
                                  error_handler=self.get_variables_error)
        
        self.network.GetVariable("thymio-II", "prox.ground.reflected", reply_handler=self.get_variables_reply_ground,
                                  error_handler=self.get_variables_error)
        
        return np.array([
                [(self.proxSensorsVal[0] / parameters.MAX_HORIZONTAL_SENSOR)],
                [(self.proxSensorsVal[1] / parameters.MAX_HORIZONTAL_SENSOR)],
                [(self.proxSensorsVal[2] / parameters.MAX_HORIZONTAL_SENSOR)],
                [(self.proxSensorsVal[3] / parameters.MAX_HORIZONTAL_SENSOR)],
                [(self.proxSensorsVal[4] / parameters.MAX_HORIZONTAL_SENSOR)],
                [(self.proxSensorsVal[5] / parameters.MAX_HORIZONTAL_SENSOR)],
                [(self.proxSensorsVal[6] / parameters.MAX_HORIZONTAL_SENSOR)],
                [(self.groundSensorsVal[0] / parameters.MAX_GROUND_SENSOR)],
                [(self.groundSensorsVal[1] / parameters.MAX_GROUND_SENSOR)]
                ])
    
    def getAllGroundSensors(self):
        self.network.GetVariable("thymio-II", "prox.ground.reflected", reply_handler=self.get_variables_reply_ground,
            error_handler=self.get_variables_error)
        
        return [(self.groundSensorsVal[0] / parameters.MAX_GROUND_SENSOR),
                (self.groundSensorsVal[1] / parameters.MAX_GROUND_SENSOR)]

    def get_variables_reply(self, r):
        self.proxSensorsVal = r

    def get_variables_reply_ground(self, r):
        self.groundSensorsVal = r

    def get_variables_error(self, e):
        print("error:")
        print(str(e))
        
    def setMotorValues(self, left, right):
        self.network.SetVariable("thymio-II", "motor.left.target", [left])
        self.network.SetVariable("thymio-II", "motor.right.target", [right])
