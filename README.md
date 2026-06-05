# Line Follow - Rescue Algorithm (Machine Learning)

Project description and technology research section

NOTE: There is no 'src' directory within this repository, instead it is called 'ml_rescue' as ROS2 packages must have the package name as the name for said directory. It has the exact same functionality as a 'src' directory.

Using vision based ML running on a Hailo NPU to rescue some balls.


Notes:

When exporting yolo model to onnx:

`yolo export model=best.pt format=onnx imgsz=640 simplify=True opset=11`

On pi, `sudo apt install dkms` `sudo apt install hailo-h10-all` and install Hailo Dataflow Compiler + Hailo Model Zoo
`hailortcli fw-control identify` to test functionality

Set up directory `calibration_images` with a few hundred images from the original dataset to calibrate hailo model.

```Bash
hailomz parse \
    --ckpt best.onnx \
    yolov8s
hailomz optimize \
    yolov8s \
    --calib-path calibration_images
hailomz compile \
    yolov8s
    --hw-arch hailo10h
```

old method:
git remote add external-origin [URL_OF_REPO]
git fetch external-origin
git merge external-origin/main --allow-unrelated-histories

better method:
git clone --mirror https://github.com/user/original-repo.git
cd original-repo.git
git push --mirror https://github.com/user/new-repo.git

Repository can be found at https://github.com/Wet-Lettuce-Robocup/toby-software-rescue