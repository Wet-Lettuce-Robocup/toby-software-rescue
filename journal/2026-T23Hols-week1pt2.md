# 1/07/2026 - 05/07/2026

### What I worked on:
Added calculations for evacuation point distance/angle using the height of the evacuation point and similar trigonometry to ball calculations. I noticed that checking contours for evacuation points took considerably longer than ml inference, which slowed down the performance of the robot, and so I will need to add green/red to the model in the future (not right now since we are at the competition in South Korea). I had to change the post-processing of inference data because the new model outputted twice the amount of data compared to the old model (1 class -> 2 classes). I also slowed down inference from 30fps to 10fps because the AI Hat drew too much power and caused the Raspberry Pi to randomly shut down.

### Reflection
The competition was pretty stressful as many parts of our code kept breaking right before we went to compete on a course. In hindsight, we needed to start coding 1-2 months earlier than we did, to give us enough time to have had something polished for the competition. Still, it was very fun and a good learning opportunity, and I took some ideas from other teams to hopefully optimise rescue for the upcoming state competition.

### Decisions made
I had to use contours to detect evacuation points since the machine learning model was not completed in time, and this impacted the framerate during rescue. I now have some ideas for future improvements, listed below in testing notes.

### Testing notes
Some ideas to improve rescue code for the future:
- Lower resolution even further, as less pixels means better performance
- AI Hat adds some latency, and it may be possible to run the machine learning model on the Pi if everything else is optimised enough
- I am curently using YOLOv8s, but the 2nd place team found that YOLOv8n performed better than the s model because it was more lightweight whilst still maintaining decent accuracy
- I could even switch to use YOLO26, which is claimed to be much faster than YOLOv8 when running on a CPU like the Raspberry Pi, and it removes NMS (Non-Maximum Suppression), further reducing latency.
- I need to take way more photos in different conditions, including random photos so that it doesn't detect clocks or faces as a ball.

### AI log use
N/A
