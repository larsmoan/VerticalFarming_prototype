# Serves as a spot to store global variable for the system

from enum import Enum
from ml_ripeness import calculate_ripeness as cr

# Used for the FSM
class State(Enum):
    IDLE = 0
    CAPTURE = 1
    PROCESSING = 2
    SAVE = 3


def fsm(state = State.IDLE):
    #Load the model
    ripeness_model = cr.load_model('ripeness_model.h5')

    while True:
        if state == State.IDLE:
            #Check for keypress
            inp = input("\n \n Currently IDLE. Press 'a' to process the test image \n \n")
            if inp == 'a':
                state = State.PROCESSING

        elif state == State.PROCESSING:
            #Process image
            print('Processing image')
            cr.process_and_predict('test.png', ripeness_model)
            #Change state to save
            state = State.IDLE