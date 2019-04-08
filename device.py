from emergent.archetypes.node import Device, Control
from utility import dev, experiment
import numpy as np
from run import cost
from parameters import atom

class VirtualDevice(Device):
    def __init__(self, name, parent, inputs):
        ''' Initializes a dummy device to store simulation states attached to
            a control node with the passed inputs. '''
        super().__init__(name, parent)
        for input in inputs:
            self.add_input(input)

class MOTControl(Control):
    def __init__(self, name, path=None):
        super().__init__(name, path=path)

    @experiment
    def capture_efficiency(self, state, params={'atoms':100, 'vmin':5, 'vmax':30}):
        self.actuate(state)
        return -cost(self.state, params=params)
