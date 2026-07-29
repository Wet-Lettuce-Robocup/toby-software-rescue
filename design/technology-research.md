# Technology Research


### Project Rationale

I will be developing a software solution for the 'rescue' component for my team's robot to compete at RoboCup Junior Rescue Line at the international competition hosted in South Korea during the June/July holidays. Essentially, the robot will need to autonomously locate two silver balls and one black ball, which represent victims stranded inside a chemical spill. It will then pick them up and deposit the silver balls into a green evacuation point, and the black ball into a red evacuation point, the position of all of these can be randomised within the evacuation area. The robot may have to avoid debris, obstacles or fake victims meant to disorient the robot. (Current competition rules found [here](https://junior.robocup.org/wp-content/uploads/2026/02/RCJRescueLine2026-final.pdf))

My solution to this challenge is to develop a machine learning model based on the popular YOLOv8, and run it on a separate NPU to boost performance. It will detect silver/black balls and the green/red evacuation points, and then a Python script will calculate how far the robot needs to move to reach those obstacles, and a separate Python script will handle the movement, sensor and servo logic.

This project provides the opportunity to develop skills that can be used in the real world, where robots are used in response to natural disasters and humanitarian relief, working in tough environments that humans would struggle in.


### Current Technologies

- **Python3 (language)** - I already know Python very well, which speeds up development. It is my go-to for any software project.
- **ROS2 (framework)** - Industry standard for robotics, allows for multi-threading and both C++/Python support. ROS2 makes it possible to have functions running asynchronously, which is amazing for what I need.
- **Raspberry Pi (hardware)** - I have plenty of experience with the Raspberry Pi family, using it in robotics since 2023. It has good processing power and plenty of GPIO pins for what we need.
- **STM32 (hardware)** - Previously we used an ESP32, but the STM32 is faster, has better peripheral control, and uses less power. It is new to the robot this year.
- **Ruff (software)** - A linting tool, used to ensure code follows style guidelines.
- **OpenCV (library)** - A powerful library for basic and advanced computer vision, it is the best at what it does and is easy to use.
- **Ultralytics YOLO (platform/framework)** - A high-performance family of real-time computer vision models, for object detection etc. Fast and efficient, and is recommended for being easy to use and train.
- **Hailo Platform (platform/software/hardware)** - They made a NPU that is compatible with the Raspberry Pi, with a suite of software to run highly optimised models on the NPU. Made for speed, but not very user friendly.
- **CVAT (software)** - Image annotation tool for model training, it is fairly easy to learn and works well for what I need.
- **Visual Studio Code (software)** - Free code editing software that I have used for six years, it has a wide range of extensions and is easily customisable and user friendly.
- **VNC Viewer (software)** - Software that provides a graphical connection to the Raspberry Pi, allowing for viewing OpenCV and camera video output easily.
- **Github (software)** - Version control software, easy to use and works well for my needs.
- **Docker (software/platform)** - Containerisation software, allows for running environments (such as Linux) on any device within a controlled environment.


### Emerging Technologies

Machine learning and artificial intelligence is still rapidly being developed, and plays a key role in my project. Machine learning, or more specifically neural networks, is where a computer learns to predict outcomes based on training data or reinforcement learning, finding patterns in data instead of having hard-coded answers. 

AI is still developing, as large corporations spend billions to improve large language models (LLMs) and autonomous agents. AI is beyond the basic testing phase, it is now being integrated into society, vastly improving some fields such as robotics and factory automation. New models are faster, cheaper to run, and are trained on more data, leading to more accurate results compared to a few years ago.

Currently, I am using a vision-based neural network that I trained to be able to detect silver/black balls, with very high accuracy. Whilst this model is good, it does not perform very well in different lighting conditions or environments, and future innovations in machine learning could improve the accuracy of my model. There could even be a much better solution to object detection that is discovered in the future, improving what is already a powerful system.


### Influence on Design

Previously, victim detection was purely camera based, using OpenCV to detect areas where a specific colour was present, which was unreliable in different lighting conditions. Using a machine learning solution (YOLO + Hailo) is a game changer, as the robot is able to adapt to new environments easier than before, and I can train the robot using actual images taken from the robot (using the Raspberry Pi), to then easily annotate in CVAT. 

Whilst the intial setup of the process was time consuming, improving and retraining the model isn't very hard to do, meaning that I can get very accurate bounding boxes around detected balls. With tight bounding boxes, post-processing to estimate (with trigonometry) the position of the ball becomes very accurate, making this process fairly reliable.

Due to using ROS2, the layout and processing of code becomes very different to what I was used to, as everything runs in nodes, and data is moved between nodes with topics, services and actions. This made the structure chart a bit complicated, as there are two nodes running completely separately from each other, transferring data between them with services. 

On one hand, this was good as machine learning inference could happen in a constant loop, unaffected by anything else. Whilst that was happening, the logic loop could occur in the other node, without being blocked by post-processing latency. On the other hand, it made programming very tricky, as I had to consciously make sure that every part of the program did not block other functions from executing.