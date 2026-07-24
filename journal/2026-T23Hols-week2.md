# 06/07/2026 - 12/07/2026

### What I worked on:
Recovering after travelling, but I started working on properly implementing my own rescue logic, instead of relying on Bain's logic which was just for the competition where we needed something to work. Added subscribers/publishers for time of flight sensors, servos, and front led. Wrote logic for grabbing balls in correct order (2 silvers then black), and readded movement class and functions for driving. 

### Reflection
Didn't really do much testing this week, as I was mainly just writing rescue logic. It was a pretty time consuming process, because I wanted to make sure that every part of rescue is non-blocking/async, whilst not being too complex. In the end, I am pretty happy with the current logic layout.

### Decisions made
Since we are using ros, the rescue logic runs in a timer set to loop every 0.1s. To make sure that it can run every 0.1s, we can't use time.sleep() or other similar processes that block the code for long periods of time. I used a state machine to switch between states when a condition is met, such as changing to TARGET_BALL when a ball is detected by the machine learning model.

### Testing notes
N/A

### AI log use
N/A