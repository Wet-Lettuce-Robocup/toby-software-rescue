# Line Follow - Rescue Algorithm (Machine Learning)

Project description and technology research section

NOTE: There is no 'src' directory within this repository, instead it is called 'ml_rescue' as ROS2 packages must have the package name as the name for said directory. It has the exact same functionality as a 'src' directory.

Using vision based ML running on a Hailo NPU to rescue some balls.

Design and develop a machine learning algorithm to be used on a robot that will compete in the International Division of Rescue Line in Robocup Junior in Incheon during the winter holidays. This machine learning algorithm will be used for the ‘rescue’ element of the course, where the robot has to autonomously locate and pick up ‘victims’, then transport them into an evacuation point, and exit the rescue zone afterwards. The ‘victims’ are 40-50mm balls which are either ‘dead’ (black, non-conductive balls) or ‘alive’ (silver, reflective, conductive balls), which will be identified using a computer vision algorithm. Additionally, I will need to create the logic for searching for the balls, navigating to the evacuation points, and navigating out of the rescue zone. 

https://codewave.com/insights/how-to-develop-a-neural-network-steps/

