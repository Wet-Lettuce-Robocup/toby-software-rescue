To build base image:
* Requires x86_64 operating system, Docker, minimum 16GB memory, and minimum 50GB of available storage

To build:
`docker build -t hailo-base:latest .`


Then to run the actual container:
`docker compose up`

Once container is built, in another terminal window run:
`docker compose exec app bash`
```bash
cd ..
. venv/bin/activate
cd app
python src/main.py
```

When exporting yolo model to onnx:
`yolo export model=best.pt format=onnx imgsz=640 simplify=False opset=12 dynamic=False`

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