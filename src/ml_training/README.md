To annotate images, I used a software called CVAT (Computer Vision Annotation Tool).

I hosted it locally on my laptop using Docker. 

When using CVAT, you can automatically annotate images using a ml (machine learning) model. How I did this:

In the directory cvat/serverless, paste the shell script and 'yolov8' directory, then navigate to `cvat/` and run:

 `docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d`
 this first option seemed to have worked, idk which one is better
` docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml -f docker-compose.override.yml up -d`

Setup is now done.

Run:
`./serverless/deploy_cpu.sh serverless/yolov8/nuclio`

If it works, the model should appear in the 'models' tab of CVAT. To make it work on mac, I had do the following:
`brew install bash`
`brew install coreutils`

If the build fails:
```zsh
docker run -d \
  --name nuclio-dashboard \
  -p 8070:8070 \
  -v /var/run/docker.sock:/var/run/docker.sock \
  quay.io/nuclio/dashboard:stable-amd64
  ```
  and go to localhost:8070 and delete the build.

After annotation is done, export to YOLO1.1.
Extract zip and delete obj.data and obj.names
Rename obj_train_data directory to labels
Place the folder containing images into the same base directory
Move labels into base labels directory from upload

Run train_val_split.py from the ml_training directory and it will copy the images and labels into train and val subfolders.

Change the values in conf.yaml to match the classes of objects that were annotated.

Run train.py, on a laptop 300 epochs takes about 3 hours to run. The training will stop early if there is no improvement for 50 epochs.


Once training is complete, find the best.pt file in runs/detect/train and run the command below:

`yolo export model=best.pt format=onnx imgsz=640 simplify=False opset=12 dynamic=False`

This will convert the PyTorch model to a .onnx model, which is now ready to be compiled into Hailo's proprietary format.
Move the .onnx model into /src/hailo_docker/models