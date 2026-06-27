When using CVAT, you can automatically annotate images using a ml model. How I did this:

In the directory cvat/serverless, paste the shell script and 'yolov8' directory, then run:

 `docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d`
` docker compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml -f docker-compose.override.yml up -d`


`./serverless/deploy_cpu.sh serverless/yolov8/nuclio`

If it works, the model should appear in the 'models' tab of CVAT. To make it work on mac, I had do the following:
`brew install bash`
`brew install coreutils`

After annotation is done, export to YOLO1.1.
Extract zip and delete obj.data and obj.names
Rename obj_train_data directory to labels
Place the folder containing images into the same base directory
Move labels into base labels directory from upload

Move train_val_split.py into directory and run it.

Move labels and images directories over to this ml_training directory and leave behind the images and labels that aren't in train/val



When exporting yolo model to onnx:
`yolo export model=best.pt format=onnx imgsz=640 simplify=False opset=12 dynamic=False`
