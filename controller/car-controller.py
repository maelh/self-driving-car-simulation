# for training: training/steering-model-training.ipynb

try:
    import vrep
except:
    print('--------------------------------------------------------------')
    print('"vrep.py" could not be imported. This means very probably that')
    print('either "vrep.py" or the remoteApi library could not be found.')
    print('Make sure both are in the same folder as this file,')
    print('or appropriately adjust the file "vrep.py"')
    print('--------------------------------------------------------------')
    print('')

# reduce console clutter
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import sys            
import math
import pygame
import numpy as np

import ctypes as ct

from aenum import IntEnum

from pathlib import Path
from itertools import chain, combinations
from time import gmtime, strftime

# PIL and pygame use the same array shape for images: (height, width, colorchannels)
#import PIL

import fastai.vision as faiv
import fastai.learner as fal
import torchvision.transforms as torchtfms

#faiv.defaults.device = 'cpu'

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def simxGetVisionSensorImage(clientID, sensorHandle, options, operationMode):
    '''
    More efficient version of vrep.simxGetVisionSensorImage, that uses numpy
    arrays to avoid copying.
    '''

    resolution = (ct.c_int*2)()
    c_image  = ct.POINTER(ct.c_byte)()    
    bytesPerPixel = 3
    if (options and 1) != 0:
        bytesPerPixel = 1
    ret = vrep.c_GetVisionSensorImage(clientID, sensorHandle, resolution,
                                      ct.byref(c_image), options, operationMode)
    c_image = ct.cast(c_image, ct.POINTER(ct.c_ubyte))

    reso = []
    image = []
    if (ret == 0):
        image = np.ctypeslib.as_array(c_image,
            shape=(resolution[0] * resolution[1] * bytesPerPixel,))
        for i in range(2):
            reso.append(resolution[i])
    return ret, reso, image    

class BasicDirection(IntEnum):
    FORWARD = 0,
    BACKWARD = 1,
    LEFT = 2,
    RIGHT = 3

    @staticmethod
    def all():
        return set([e for e in BasicDirection])

def steering_to_str(steering: set(BasicDirection)):
    if steering == { BasicDirection.FORWARD }:
        return 'Forward'
    elif steering == { BasicDirection.FORWARD, BasicDirection.LEFT }:
        return 'ForwardLeft'
    elif steering == { BasicDirection.FORWARD, BasicDirection.RIGHT }:
        return 'ForwardRight'
    return ''
    #result = ', '.join(str(e) for e in steering)
    #if result == '':
    #    result = '()'
    #return result

class AckermannDriving:

    def __init__(self):
        super().__init__()

        self._FORWARD_VELOCITY = 300*math.pi/180
        self._TURNING_ANGLE = 45*math.pi/180

        self._currentSteering = set()

        self._steer_detect = False
        self._auto_drive = False

        self._recording_start = None

        self._frameNo = {}
        for subset in powerset(set(BasicDirection)):
            self._frameNo[steering_to_str(set(subset))] = 0

        #learner = cnn_learner(data, models.resnet18, pretrained=True, metrics=accuracy)

        # steady left turn
        #path = Path.cwd() / 'training_data' / '2019-10-15 20.21.24' # '2019-10-11 20.46.46'
        #self._learner = fal.load_learner(path, 'street_vrep_squared_google_collab.pkl')

        # steady left turn
        #path = Path.cwd() / 'training_data' / '2019-10-15 20.21.24'
        #self._learner = fal.load_learner(path, 'street_vrep_squared.pkl')

        # unsteady left turn
        #path = Path.cwd() / 'training_data' / '2019-10-16 19.07.23'
        #self._learner = fal.load_learner(path, 'street_vrep_squared_unsteady.pkl')       

        # steady right turn
        #path = Path.cwd() / 'training_data' / '2019-10-16 20.01.39 - perfect right turn'
        #self._learner = fal.load_learner(path, 'street_vrep_perfect_right_turn.pkl')    
        
        # steady left and right turn
        #path = Path.cwd() / 'training_data' / 'perfect left and right turn'
        #self._learner = fal.load_learner(path, 'street_vrep_perfect_left_and_right_turn.pkl')  
        
        # steady left and right turn (from short and long road end)
        #path = Path.cwd() / 'training_data' / 'perfect left and right turn, from short and long end' / 'street_vrep_perfect_left_and_right_turn_short_and_long_side2.pkl'
        
        path = Path.cwd() / 'models' / 'perfect left and right turn, from short and long end.pkl'
        self._learner = fal.load_learner(path)

    def _vrep_init(self):
        # just in case, close all opened connections
        vrep.simxFinish(-1)

        # Connect to V-REP
        self._clientID = vrep.simxStart("127.0.0.1", 19999, True, True, 1000, 5)
        result = self._clientID != -1

        if result:
            print("Connected to remote API server")

            vrep.simxSynchronous(self._clientID, True)

            self._vrres, self._steeringLeft = vrep.simxGetObjectHandle(
                self._clientID, 'car_steeringLeft',
                vrep.simx_opmode_blocking)
            self._vrres, self._steeringRight = vrep.simxGetObjectHandle(
                self._clientID, 'car_steeringRight',
                vrep.simx_opmode_blocking)
            self._vrres, self._motorLeft = vrep.simxGetObjectHandle(
                self._clientID, 'car_motorLeft',
                vrep.simx_opmode_blocking)
            self._vrres, self._motorRight = vrep.simxGetObjectHandle(
                self._clientID, 'car_motorRight',
                vrep.simx_opmode_blocking)

            self._vrres, self._visionSensor = vrep.simxGetObjectHandle(
                self._clientID, 'car_visionSensor',
                vrep.simx_opmode_blocking)

            self._vrres, self._visionSensorRes, image_raw = \
                simxGetVisionSensorImage(self._clientID, self._visionSensor, 0,
                vrep.simx_opmode_blocking)

        self._vrres = vrep.simx_return_ok

        return result

    def _vrep_final(self):
        if self._clientID != -1:
            # Before closing the connection to V-REP, make sure that the last
            # command sent out had time to arrive. You can guarantee this with
            # (for example):
            vrep.simxGetPingTime(self._clientID)

            # Now close the connection to V-REP:
            vrep.simxFinish(self._clientID)
        else:
            print("Not connected to remote API server")

    def _pygame_init(self):
        pygame.init()
        self._scaling = 8
        self._screen = pygame.display.set_mode((
            self._visionSensorRes[0] * self._scaling,
            self._visionSensorRes[1] * self._scaling))
        pygame.display.set_caption("Car Camera and Ctrl")

    def _pygame_loop(self):
        running = True
        imageidx = 0
        while running:
            if self._vrres != vrep.simx_return_ok and \
                vrep.simxGetConnectionId(self._clientID) == -1:
                self._vrep_init()

            self._vrres, resolution, image_raw = simxGetVisionSensorImage(
                self._clientID, self._visionSensor, 0,
                vrep.simx_opmode_continuous)

            if self._vrres == vrep.simx_return_ok:
                bytelen = len(image_raw)
                image = image_raw.reshape(resolution[1], resolution[0], 3)

                # v-rep array flipped compared to pygame and PIL image arrays
                image = np.flipud(image)

                surface = pygame.image.frombuffer(image.reshape(bytelen),
                    resolution, "RGB")                            
                self._save_training_data(surface)

                surface = pygame.transform.scale(surface, 
                    (resolution[0] * self._scaling,
                    resolution[1] * self._scaling))
                self._screen.blit(surface, (0, 0))

                # TODO: smaller camera pictures/frames
                # TODO: synchronize VREP progress with reactiveness of cnn-predictor
                # TODO: ensure fixed time steps between frames, so that the car always drives for dt time in a certain steering direction

                if self._steer_detect:                    
                    image = image.copy() # fix negative stride
                    tensor = image
                    #tensor = torchtfms.ToTensor()(image)
                    #print(tensor)
                    
                    # TODO: need to normalize?
                    # Mean subtraction, Scaling by some factor
                    # or does learner.predict() do this already?
                    # https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/

                    # TODO: slowest part is the prediction (that's why we need to synchronize with V-REP so it does not miss frames and therefore does not delay steering; but this wont work on the real robot car)
                    prediction = self._learner.predict(tensor)

                    predictedCategory = str(prediction[0])

                    if predictedCategory == "Forward":
                        steerCmd = { BasicDirection.FORWARD }
                    elif predictedCategory == "ForwardLeft":
                        steerCmd = { BasicDirection.FORWARD, BasicDirection.LEFT }
                    elif predictedCategory == "ForwardRight":
                        steerCmd = { BasicDirection.FORWARD, BasicDirection.RIGHT }
                    else:
                        steerCmd = set()

                    predictedCategoryProbability = float(max(prediction[2]))
                    
                    predictionMessage = '{} \t(prob={:.2%})'.format(
                        predictedCategory, predictedCategoryProbability)
                    print(predictionMessage)

                    self._displaySteeringPrediction(predictedCategory,
                        predictedCategoryProbability)

                    if self._auto_drive:
                        self._drive_car(steerCmd)

            vrep.simxSynchronousTrigger(self._clientID)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_s:
                        self._reset_car(1)
                    elif event.key == pygame.K_w:
                        self._reset_car(2)
                    elif event.key == pygame.K_e:
                        self._reset_car(3)
                    elif event.key == pygame.K_q:
                        self._reset_car(4)
                    elif event.key == pygame.K_r:
                        self._reset_car(5)
                    elif event.key == pygame.K_SPACE:
                        self._print_car_pos()
                    elif event.key == pygame.K_d:
                        self._toggle_steer_detect()
                    elif event.key == pygame.K_a:
                        self._toggle_auto_drive()

                    if not self._auto_drive:
                        if event.key == pygame.K_UP:
                            self._steer_car({BasicDirection.FORWARD})
                        elif event.key == pygame.K_DOWN:
                            self._steer_car({BasicDirection.BACKWARD})
                        elif event.key == pygame.K_LEFT:
                            self._steer_car({BasicDirection.LEFT})
                        elif event.key == pygame.K_RIGHT:
                            self._steer_car({BasicDirection.RIGHT})
                                      
                elif event.type == pygame.KEYUP:
                    if not self._auto_drive:
                        if event.key == pygame.K_UP:
                            self._unsteer_car({BasicDirection.FORWARD})
                        elif event.key == pygame.K_DOWN:
                            self._unsteer_car({BasicDirection.BACKWARD})
                        elif event.key == pygame.K_LEFT:
                            self._unsteer_car({BasicDirection.LEFT})
                        elif event.key == pygame.K_RIGHT:
                            self._unsteer_car({BasicDirection.RIGHT})

            pygame.display.flip()  
            
    def _displaySteeringPrediction(self, predictedCategory, predictedCategoryProbability):
        emptyBuff = bytearray()
        predictedCategoryProbabilityStr = '(prob={:.2%})'.format(predictedCategoryProbability)
        self._vrres, retInts, retFloats, retStrings, retBuffer = \
            vrep.simxCallScriptFunction(self._clientID,
                'remoteApiCommandServer', vrep.sim_scripttype_childscript,
                'displaySteeringPrediction_function', [],
                [predictedCategoryProbability],
                [predictedCategory, predictedCategoryProbabilityStr], emptyBuff,
                vrep.simx_opmode_blocking)        
            
    def _reset_car(self, reset_index):
        #vrep.simxPauseSimulation(self._clientID, vrep.simx_opmode_oneshot)

        emptyBuff = bytearray()
        self._vrres, retInts, retFloats, retStrings, retBuffer = \
            vrep.simxCallScriptFunction(self._clientID,
                'remoteApiCommandServer', vrep.sim_scripttype_childscript,
                'resetCar' + str(reset_index) + '_function', [], [], [], emptyBuff,
                vrep.simx_opmode_blocking)

        self._currentSteering = set()

        self._steer_detect = False
        self._auto_drive = False
        
        self._recording_start = None
        
        self._frameNo = {}
        for subset in powerset(set(BasicDirection)):
            self._frameNo[steering_to_str(set(subset))] = 0

        #vrep.simxStartSimulation(self._clientID, vrep.simx_opmode_oneshot)

    def _print_car_pos(self):
        #vrep.simxPauseSimulation(self._clientID, vrep.simx_opmode_oneshot)

        emptyBuff = bytearray()
        self._vrres, retInts, retFloats, retStrings, retBuffer = \
            vrep.simxCallScriptFunction(self._clientID,
                'remoteApiCommandServer', vrep.sim_scripttype_childscript,
                'printCarPos_function', [], [], [], emptyBuff,
                vrep.simx_opmode_blocking)

        self._currentSteering = set()

        self._steer_detect = False
        self._auto_drive = False
        
        self._recording_start = None

        self._frameNo = {}
        for subset in powerset(set(BasicDirection)):
            self._frameNo[steering_to_str(set(subset))] = 0

        #vrep.simxStartSimulation(self._clientID, vrep.simx_opmode_oneshot) 

    def _toggle_steer_detect(self):
        self._steer_detect = not self._steer_detect

    def _toggle_auto_drive(self):
        self._auto_drive = not self._auto_drive
        if not self._auto_drive:
            self._unsteer_car(self._currentSteering)

    def _set_car_wheel_velocity(self, wheelVelocity):
        vrep.simxSetJointTargetVelocity(self._clientID, self._motorLeft,
            wheelVelocity, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(self._clientID, self._motorRight,
            wheelVelocity, vrep.simx_opmode_blocking)

    def _set_car_steering_angle(self, steeringAngle):
        if steeringAngle == 0:
           steeringAngleLeft = 0
           steeringAngleRight = 0
        else:
            d = 0.755 # 2*d=distance between left and right wheels
            l = 2.5772 # l=distance between front and read wheels
            steeringAngleLeft=math.atan(l/(-d+l/math.tan(steeringAngle)))
            steeringAngleRight=math.atan(l/(d+l/math.tan(steeringAngle)))

        vrep.simxSetJointTargetPosition(self._clientID, self._steeringLeft,
            steeringAngleLeft, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetPosition(self._clientID, self._steeringRight,
            steeringAngleRight, vrep.simx_opmode_blocking)

    def _drive_car(self, steerCmd: set(BasicDirection)):
        self._unsteer_car(self._currentSteering - steerCmd)

        # do not activate steering, that was activated already
        steerCmd = steerCmd - self._currentSteering

        if BasicDirection.FORWARD in steerCmd:
            self._set_car_wheel_velocity(self._FORWARD_VELOCITY)

        if BasicDirection.BACKWARD in steerCmd:
            self._set_car_wheel_velocity(-self._FORWARD_VELOCITY)

        if BasicDirection.LEFT in steerCmd:
            self._set_car_steering_angle(self._TURNING_ANGLE)

        if BasicDirection.RIGHT in steerCmd:
            self._set_car_steering_angle(-self._TURNING_ANGLE)

        self._currentSteering = self._currentSteering | steerCmd

    def _steer_car(self, steerCmd: set(BasicDirection)):
        # do not activate steering, that was activated already
        steerCmd = steerCmd - self._currentSteering

        if BasicDirection.FORWARD in steerCmd:
            self._set_car_wheel_velocity(self._FORWARD_VELOCITY)

        if BasicDirection.BACKWARD in steerCmd:
            self._set_car_wheel_velocity(-self._FORWARD_VELOCITY)

        if BasicDirection.LEFT in steerCmd:
            self._set_car_steering_angle(self._TURNING_ANGLE)

        if BasicDirection.RIGHT in steerCmd:
            self._set_car_steering_angle(-self._TURNING_ANGLE)

        self._currentSteering = self._currentSteering | steerCmd

        if BasicDirection.FORWARD in self._currentSteering:
            self._start_recording()

    def _unsteer_car(self, steerCmd: set(BasicDirection)):
        # do not deactivate steering that was deactivated already
        steerCmd = steerCmd - (BasicDirection.all() - self._currentSteering)

        if BasicDirection.FORWARD in steerCmd:
            self._set_car_wheel_velocity(0)

        if BasicDirection.BACKWARD in steerCmd:
            self._set_car_wheel_velocity(0)

        if BasicDirection.LEFT in steerCmd:
            self._set_car_steering_angle(0)

        if BasicDirection.RIGHT in steerCmd:
            self._set_car_steering_angle(0)

        self._currentSteering = self._currentSteering - steerCmd

        if not (BasicDirection.FORWARD in self._currentSteering):
            self._stop_recording()

    def _is_recording(self):
        return self._recording_start != None

    def _start_recording(self):
        if not self._is_recording():
            self._recording_start = strftime("%Y-%m-%d %H.%M.%S", gmtime())

    def _stop_recording(self):
        if self._is_recording():
            self._recording_start = None

    def _save_training_data(self, current_frame):
        if not self._is_recording():
            return

        currentSteeringStr = steering_to_str(self._currentSteering)

        if currentSteeringStr == '':
            return

        frameNo = self._frameNo[currentSteeringStr]

        path = Path.cwd() / Path('training_data')
        path = path / self._recording_start / currentSteeringStr
        path = path / (self._recording_start + ' - ' + str(frameNo) + '.png')
        path.parent.mkdir(parents = True, exist_ok = True)
        pygame.image.save(current_frame, str(path))

        self._frameNo[currentSteeringStr] = frameNo + 1

    def run(self):
        if self._vrep_init():
            self._pygame_init()
            try:
                self._pygame_loop()
            finally:
                pygame.quit()

        self._vrep_final()

if __name__ == '__main__':
    ackermannDriving = AckermannDriving()
    ackermannDriving.run()
