To build base image:
* Requires x86_64 operating system, Docker, minimum 16GB memory, and minimum 50GB of available storage

Make sure you have the Hailo Dataflow Compiler .whl installed (eg. hailo_dataflow_compiler-5.3.0-py3-none-linux_x86_64.whl), which can be downloaded from the Hailo Developer Zone on their website, and move the file into the main hailo_docker dir.

Copy over the training images into /calibration_images
Make sure that the config in /src matches the amount of classes that your model has, and modify the main.py script if needed.

To build the Dataflow Compiler:
`docker build -t hailo-base:latest .`

Then to run the actual container:
`docker compose up`

Eventually, it should output a .hef file in the /models directory. This is the file that can run on the Raspberry Pi AI Hat+ 2.


On pi, `sudo apt install dkms` and install the files from their website, including the pcie driver and hailort python wheel.

<!-- `hailortcli fw-control identify` to test functionality -->
