# SelfLocalisationCF
This project aims to implement self localisation using a grid cell model as presented by guanella et al (doi:10.1142/S0129065707001093) in the crazyflie drone as simulated in the webots environment (Details:https://www.bitcraze.io/documentation/tutorials/getting-started-with-simulation/)

## Files and Folders:
 - SelfLocalisationCF.py
	- This is the main script and control flow of the program.
- GridNetwork.py
	- This code implements the grid cell model of the guanella paper (doi:10.1142/S0129065707001093). Thanks to Raimon Bullich Villareal for providing the bulk of the code.
- DroneController.py
	- Implements a controller unit that is used to control the drone in webbots. Makes use of a PID controller. Sourced and modified sample controller code.
- Sample
	- Within this folder are the sample scripts of webots for the wall_following drone controller._ 
- Results
	- In this folder are the most recent plots and results
- Deprecated
	- This folder features old version of scripts which are not used any more.
- SquareBox.wbt
	- this is the world file used by webots to construct the environment.
- runSLC.bat
	- A batch file that can be run in the cmd to start the controller in webots

## Latest Change:
**13.03.:** Added a method that produces an overview of mse scores for a sequence of gains that are tested. The progam crashed with a BSOD once so care should be applied when trying to run it. The best results are generally achieved with a high number of gains (5-6) and low spacing between them (0.1-0.2). Spacing here means the difference between two adjacend gains within the same list (e.g. spacing=0.2, gains = [0.2, 0.4, 0.6, etc]). The spacing seems to have a higher impact and mse scores generally increase with longer simulation times.<br>
**07.03.:** Tweaks on how results are saved. Added the possibility to repeat the simulation multiple times for different sets of gain parameters<br>
**07.03.:** Added documentation for the GridNetwork file. Introduced a new function to generate and save the activity plot for each grid cell neuron.<br>
**06.03.:** Wrote a tutorial for setting up the project. Introduced helper functions for saving and loading data. Tested out different configurations of gain parameters to achieve high predictions.<br>
**05.03.:** Reworked the main script SelfLocalisationCF.py and added documentation. Tweaked the random movement algorithm to use small angle adjustment each step.<br>
**04.03.:** Introduced a new way to achieve random movement. It is based on Brownian motion and utilises the Ornstein-Uhlenbeck process.<br>
**04.03.:** Fixed an issue when passing movement comments to the controller. The controller works with body-centred coordinates (forward, sideways), but the main control calculated new directions in absolute position(x, y). The controller now correctly transforms the input into body centred coordinates<br>
**28.02.:** Introduced a new function that adds a drift towards the origin to new generated translational movement. It has radial dependency.<br>
**28.02.:** Worked on the plotting functions to discern error sources, generalised the code<br>
**28.02.:** Implemented a new algorithm to generate random walking. Also added boundary detection and redirection, but this is still not working optimal every time<br>
**27.02.:** Debugged the program. It finally executes completely without errors and produces the two plots at the end.<br>
**27.02.:** Changed some lines in the GridNetwork code, which were sources of overflow errors.<br>
**26.02.:** Included the changes added by raimon to decode the location. Also moved the PID code to the controller, which deprecates the pid_controller file, hopefully reducing the complexity of the project.<br>
**15.02.:** Added code to update the grid network on every time step. It seems that the network is prone to overflow issues after just a couple of time steps(~20sec).<br>
**15.02.:** Corrected the yaw behaviour. Previously the yaw of the drone would be oscillating between two values. It now correctly stays in place until a new rotation command is given. The drone now stably moves for large simulated times (>1h), but seems to have a preference of staying within a certain quadrant.


## Current Tasks (no particular order):
1. The project is getting long and complicated codewise. Proper documentation is crucial to maintain overview. Improve the readability and documentation of the code.
1. Revert the project to run only a single simulation using the best performing set of gains
3. Add altitude control to the random walk and GridNetwork.
4. Test the feasibility and processing time of path prediction during simulation.
1. Plot the prediction error over time if possible to investigate time-dependency.
