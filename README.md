# Simple reproducible self-driving car

Realizing even a simple self-driving car is a non-trivial project, when hardware is involved.

This self-driving car project uses the simulator CoppeliaSim (V-REP) to present a fully software based solution.
It provides:
  - Simple noise free and high contrasting camera pictures
    - Allows for easier experimenting with deep learning, while not getting distracted with the complexity of a highly dynamic real world (shadows, changing sun or internal lighting, shaking of the camera while driving, lens distortion, noise from the camera sensor, and multiple other possible effects that need to be filtered out or compensated for).
      - In the real world, the need to create a training set that captures all this variability in sufficient detail, makes it difficult to get started, and less easy to debug issues, since results are less predictable.
  - A predefined test track that allows to test the exact same situation again and again
    - You can always add variability later on, such as randomizing the starting position or adapting tracks. But when you get started you can be sure everything works as intended, without experimenting.
  - A self-driving mode that is guarantueed to work, since the environment is the same as the one it was trained on.
  - Simple capturing of training data by driving the car in the simulator using your arrow keys.
  - A Jupyter notebook to fit the CNN (convolutional neural network) on the captured training data.
  - Simple code that focuses on the bare essentials.
  - Complete set of steps that can be followed to have the same result, again, without unknown factors getting in the way.

## Hardware alternatives

Usual approaches are to modify a toy RC car, by adding a small computer, a camera, a battery pack, and a way to drive the motors. This means you need to either hack the remote control, or add electronics to drive the motors using a microcontroller. You also have to pick a suited camera. It turns out that wide angle view cameras are necessary to get a sufficiently descriptive view of the road to drive along.

Single-board computers like a Raspberry Pi are usually not powerful enough to derive steering commands with low enough latency from camera input. Jetson Nano is a good alternative since it provides accelerated inference over its GPU.

If you have a 3d printer and dont mind buying various components, you can look into [JetRacer](https://github.com/NVIDIA-AI-IOT/jetracer) or buy a complete kit from WaveShare. Another option is [Donkey Car](https://github.com/autorope/donkeycar), but again a 3d printer is helpful, since distributors are not available in every country.

While those kits help, they are not without issues, and require enough familiarity with hardware to be able to troubleshoot problems. A simulator allows to try out various hardware setups more easily, by tuning camera or motor properties, and more easily creating test tracks.

When you are also a beginner in machine learning, a complete software package that gets you started directly, with a reproducible setup, will make it a lot easier to experiment and focus on the task at hand compared to the solutions presented above.
