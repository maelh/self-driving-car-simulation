## Install

1. Install [Anaconda](https://www.anaconda.com/products/individual)
2. Open an "Anaconda Prompt" and
    1. Install FastAI
        - either by entering `conda install -c fastai fastai`
        - or if that fails, enter `pip install fastai`
    2. Install PyGame
        - enter `pip install pygame`
    3. Install aenum
        - enter `pip install aenum`
3. Install [CoppeliaSim](https://www.coppeliarobotics.com/downloads)
    - the EDU version if free for non-commercial purposes; you can also use the Player version, but it wont be able to edit the scene
4. Clone the repository https://github.com/maelh/self-driving-car-simulation

## Run

1. Open `simulation/ackermann-steering-car-scene.ttt` in CoppeliaSim and run the scene
    - press "Simulation|Start simulation" or the play button to run the scene
2. In an opened "Anaconda Prompt" enter:
    1. `cd controller`
    2. `python car-controller.py`

On success, you should see an additional window with the title "Car Camera and Ctrl" pop up. While this window is focused, you can drive the car using the arrow keys,
enable/disable steering detection with the `d` key, engage/disengage autonomous driving with `a`, and quit using the `ESC` key.

When autonomous driving mode is not engagend, training data is recorded as soon as the car drives in a forward, left-forward, or right-forward direction,
but not when it drives backwards. Camera frames are stored in the `training/training_data/<current data and time>` directory. The sub directories are the labels of the
image files they contain, e.g., `ForwardLeft` contains all the camera frames captured when driving in the left-forward direction, i.e., holding the left and forward keys
pressed.

## Train
