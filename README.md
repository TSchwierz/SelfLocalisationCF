# SelfLocalisationCF
This projects aims to implements self localisation in the crazyflie drone as simulated in the webots envinornment (Details:https://www.bitcraze.io/documentation/tutorials/getting-started-with-simulation/)

Files:
 - SelfLocalisationCF.py
	- This is the main script and controlflow of the program.
- CFController.py
	- Implements a controller unit that is used to control the drone in webbots. Makes use of pid_controller. Sourced and modified sample controller code.
- pid_controller.py
	- [DEPRECATED: consolidated with CFController] Implementation of a fixed height pid controller for the crazyflie. Sourced from sample controller under MIT license.
- GridNetwork.py
	- This code implements the grid cell model of the guanella paper (doi:10.1142/S0129065707001093). Thanks to Raimon Bullich Villareal for providing the bulk of the code.
- Sample
	- Within this folder are the sample scripts of webots for the wall_following drone controller._ 
- SquareBox.wbt
	- this is the world file used by webots to construct the environment.
- runSLC.bat
	- A batch file that can be run in the cmd to start the controller in webots

# Latest Change:
27.02.: Debuged the program. It finally executes completly without errors and produces the two plots at the end.
27.02.: Changed some lines in the GridNetwork code, which were sources of overflow errors.<br>
26.02.: Included the changes added by raimon to decode the location. Also moved the pid code to the controller, which depricates the pid_controller file, hopefully reducing the complexity of the project.<br>
15.02.: Added code to update the grid network on every timestep. It seems that the network is prone to overflow issues after just a couple of timesteps(~20sec).<br>
15.02.: Corrected the yaw behaviour. Previously the yaw of the drone would be oscillating between two values. It now correctly stays in place until a new rotation command is given. The drone now stabily moves for large simulated times (>1h), but seems to have a preferrence of staying within a certain quadrand.


Current Tasks (no particular order):
1. when generating a new direction, check for proximity to walls using gps and create a heavy bias for direction pointing away from them.
2. generate proper figures after (during) execution of the simulation
3. change random walk generation. Every (other) second the drone should either do a translation, rotation or change in altidute, decided by random chance.
4. The plot generated to show the network activity is rather empty. This could be because of limited time spend in a big environment. check this.
1. The project is getting long and complicated codewise. Proper documentation is crucial to maintain overview. Improve the readability and documentation of the code
