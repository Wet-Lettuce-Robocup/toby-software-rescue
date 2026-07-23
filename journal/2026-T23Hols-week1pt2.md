# 1/07/2026 - 05/07/2026

### What I worked on:
2-3 sentences summarising what you built, designed or researched this week
Added calculations for evacuation point distance/angle using the height of the evacuation point and similar trigonometry to ball calculations. I noticed that checking contours for evacuation points took considerably longer than ml inference, which slowed down the performance of the robot, and so I will need to add green/red to the model in the future (not right now since we are at the competition in South Korea). I had to change the post-processing of inference data because the new model outputted twice the amount of data compared to the old model (1 class -> 2 classes). I also slowed down inference from 30fps to 10fps because the AI Hat drew too much power and caused the Raspberry Pi to randomly shut down.

### Reflection
~50 words on the most instructive moment — what went wrong first, how you fixed it, or what made progress easier

### Decisions made
Design or implementation choices you settled on, and your reasoning

### Testing notes
Any testing you performed and what you found

### AI log use
For every AI session: tool used, purpose, output received, and how you evaluated and adapted it
