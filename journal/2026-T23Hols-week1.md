# 25/06/2026 - 30/07/2026

### What I worked on:
Preparation for competition and the competition itself occured this week. For the competition, I added a function to detect green and red evacuation points by looking for those coloured contours that have an area larger than a specified threshold. This was done because I didn't have green/red evacuation point detection in the machine learning model ready in time for competition. These detections were published using the same Detection2DArray as what the balls were published to. I also finished gathering images for the new model, annotated them automatically in CVAT using the first iteration of the ml model, trained the model, and compiled it into a .hef file. Also made a script to automatically split training data into 80% training and 20% validation (both images and their labels).

### Reflection
I wanted to speed up annotation (I had over 500 images to go through) by using my previous model to annotate the images, which CVAT supported. Unfortunately, this process was quite complicated and took a lot of trial and error to get working, and didn't end up saving much time. I adapted example code found in the CVAT repository to my script, which ended up working well enough.

### Decisions made
I wanted to use CVAT auto-annotation to speed up the annotation process, which can take hours when annotating hundreds of images by hand, especially when I may be training 2 or 3 more iterations of the model. It wasn't very easy to do, and took way too long to figure out, but I hope it is worth it in the future.

### Testing notes
N/A

### AI log use
Google AI for debugging CVAT auto-annotation and for finding their documentation on it.
