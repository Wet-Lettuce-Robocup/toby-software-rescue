# Line Follow - Rescue Algorithm (Machine Learning)

Using a custom-trained vision based machine learning model running on a Hailo NPU to detect objects during rescue, competing in Robocup Junior Rescue Line.

All code is within the 'src' directory, including ros packages, code relating to hailo compilation, and code relating to machine learning model training.

I highly recommend not reading the raw markdown files, instead view the markdown files through github or another program that has built in markdown parsing.

### Note: for mirroring repository for software task
```bash
git clone --mirror https://github.com/user/original-repo.git
cd original-repo.git
git push --mirror https://github.com/user/new-repo.git
```

\* Repository can be found at [https://github.com/Wet-Lettuce-Robocup/toby-software-rescue](https://github.com/Wet-Lettuce-Robocup/toby-software-rescue)

### Camera Calibration
run `xhost +local:docker` in VNC on the raspberry pi
```
ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025 --no-service-check --ros-args -r camera:=/front_camera/camera_node/camera_info -r image:=/front_camera/camera_node/image_raw

when exec'ed into the docker container
```