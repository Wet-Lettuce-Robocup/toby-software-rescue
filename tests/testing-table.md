### Test #1 - First ML model
Testing to see if it can detect silver balls.

|Input|Expected Result|Actual Result|Status|
|---|:---:|:---:|---|
|Image of silver ball inside test area|Image annotated with silver ball surrounded with a red box|Image annotated with silver ball and power bank surrounded with a red box, silver ball reporting 30% confidence|Pass|

### Test #2 - Second ML model + video feed
Slightly changed training process compared to first model, should improve accuracy. Inputting video feed and live cv2 video output.

|Input|Expected Result|Actual Result|Status|
|---|:---:|:---:|---|
|Video stream from front robot camera, two silver balls in frame|Both balls with annotated bounding box, no matter where they are in the frame|Both balls and a clock with annotated bounding box, most of the time, with high confidence|Pass|

### Test #3 - Second ML model + video feed within ROS
Attempting to feed a video into the model within ROS environment (saving to a video file instead of live video output).

|Input|Expected Result|Actual Result|Status|
|---|:---:|:---:|---|
|Video stream from front robot camera, one silver ball in frame|Silver ball with red bounding box and confidence annotated onto image|Silver ball annotated some of the time, doesn't work very far away/up close. Averaging 80% confidence|Pass|

### Test #4 - Testing async inference
Testing how Hailo async inference functions, and whether the data output is delayed or not.

|Input|Expected Result|Actual Result|Status|
|---|:---:|:---:|---|
|Video stream from front robot camera, one silver ball in frame|Successful detection of ball and better performance than last test|No detections outputted|Fail|

Likely an issue with the code rather than the model.

### Test #5 - Retesting async inference
Added some fixes to address last test's failure.

|Input|Expected Result|Actual Result|Status|
|---|:---:|:---:|---|
|Video stream from front robot camera, one silver ball in frame|Successful detection of ball, less latency than synchronous inference|Ball successfully detected, however no drastic increase in performance|Pass|

### Test #6 - Testing sending inference detections with topic in ROS
Testing if my code for sending inference detections via a topic with type Detection2DArray will work and initial distance estimations.

|Input|Expected Result|Actual Result|Status|
|---|:---:|:---:|---|
|Video stream from front robot camera, one silver ball in frame at multiple positions in the frame|Successful inference, post-processing of data, sending of data to ml_rescue_node and somewhat accurate distance/angle estimation|Data successfully sent to ml_rescue_node with minimal latency, however distance estimation inaccurate at large angles: 200mm at 45º reading as 140mm at 77º|Pass|

### Test #7 - Distance/angle estimation after camera calibration
Calibrated the camera to fix lens distortion, now testing to see if that helped.

|Input|Expected Result|Actual Result|Status|
|---|:---:|:---:|---|
|Video stream from front robot camera, one silver ball in frame at both small and large angles from centre of frame|Distance and angle estimations accurate at any position|Distance and angle estimations much better than last test, still occasionally out by ±30mm|Pass|

### Test #8 - Third ML model - silver vs black
Testing new model (trained with 500 images) however compiling process was rushed, so accuracy may be affected.

|Input|Expected Result|Actual Result|Status|
|---|:---:|:---:|---|
|Video stream from front robot camera, one silver ball and one black ball|Both balls detected correctly with reasonable confidence|Balls correctly detected with 50% accuracy, occasionally not being detected at certain angles|Pass|

### Test #8 - Third ML model - YOLO vs Hailo
Testing model pre-compiling to see if it was training or compiling where accuracy was lost.

|Input|Expected Result|Actual Result|Status|
|---|:---:|:---:|---|
|Video stream from front robot camera, one silver ball and one black ball|Both balls detected with same confidence as the compiled Hailo model|Balls were detected with much higher confidence (80%) than Hailo model, although some locations (in a dark corner) were not detected with same level of confidence|Fail|

Hailo model lost accuracy after compiling, will have to look into that to see what went wrong (likely a value related to NMS).