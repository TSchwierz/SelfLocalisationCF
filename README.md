# SelfLocalisationCF
This projects aims to implements self localisation in the crazyflie drone as simulated in the webots envinornment (Details:https://www.bitcraze.io/documentation/tutorials/getting-started-with-simulation/)

Files:
 - SelfLocalisationCF.py
	- This is the main script and controlflow of the program.
- CFController.py
	- Implements a controller unit that is used to control the drone in webbots. Makes use of pid_controller. Sourced and modified sample controller code.
- pid_controller.py
	- Implementaion of a fixed height pid controller for the crazyflie. Sourced from sample controller under MIT license.
- GridNetwork.py
	- This code implements the grid cell model of the guanella paper (doi:10.1142/S0129065707001093). Thanks to Raimon Bullich Villareal for providing the majority of the code.


Current Tasks:
1. Add a method to determin distance to walls
2. when generating a new direction, check for proximity to walls and create a heavy bias for direction pointing away from them.
3. combine the pid and controller scripts into one module
4. Add the grid functionality to the drone