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
- Sample
	- Within this folder are the sample scripts of webots for the wall_following drone controller._ 

# Latest Change:
15.02.: Added code to update the grid network on every timestep. It seems that the network is prone to overflow issues after just a couple of timesteps(~20sec).<br>
15.02.: Corrected the yaw behaviour. Previously the yaw of the drone would be oscillating between two values. It now correctly stays in place until a new rotation command is given. The drone now stabily moves for large simulated times (>1h), but seems to have a preferrence of staying within a certain quadrand.


Current Tasks:
1. when generating a new direction, check for proximity to walls using gps and create a heavy bias for direction pointing away from them.
2. combine the pid and controller scripts into one module
3. Add raimons changes for location decoding to the network module
4. Fix Overflow issues with the network code
