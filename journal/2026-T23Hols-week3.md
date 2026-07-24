# 13/07/2026 - 19/07/2026

### What I worked on:
Continued to work on logic for rescue, including targetting and grabbing balls. Ran a quick test on the robot, fixed up some small errors. I also changed inference to only happen when it is requested using a service, to hopefully improve performance since it does not need to occur at a set fps, and added support for green/red detection when retrain the model.

### Reflection
Same as last week, just working on adding logic and trying to ensure that everything is non-blocking, which is hard when rescue code is usually procedural.

### Decisions made
I would retrain the model with more photos and annotate evacuation points, but I don't have that much time before I have exams, so I will leave that part out until the trials are done.

### Testing notes
N/A

### AI log use
N/A
