# Setup
This project and tutorial has been written with a Windows 10 implementation in mind. In order to use this project a installiation of webots and the ability to run python on your machine is required.

## Webots
Under https://cyberbotics.com you can download the latest release of webots, a simulation envirnoment to work on robotics.

Once it is installled and opened navigate to File/Open World and select the world file 'SquareBox.wbt' that is part of this repository. 
## Environmental Path Variables
In order to run the project the crazyflie drone in webots needs to be targeted by an extern controller, a python script in this case. This can be done by passing the script to the webots-controller executiable in the cmd.
```
webots-controller.exe controller.py
```
A tutorial on that can be found [here](https://cyberbotics.com/doc/guide/running-extern-robot-controllers?tab-os=windows). 

To make this easier it is recommanded to create some path variables. In windows go to settings and search for 'environmental path variables'. Add the following three variables:
- WEBOTS : pointing towards the folder of 'webots-controller.exe'. Usally this is C:\Program Files\Webots\msys64\mingw64\bin
- SLController : pointing to the local copy of the repository containing the file 'SelfLocalisationCF.py'.
- specHome : This points to the folder from which the batch script 'runSLC.bat' is run.

## Python
In order to execute the code your machine needs to be able to run python. In my case I setup a virtual envirnment using [anaconda](https://www.anaconda.com) using the following command in the anaconda prompt:
```
conda create --name myenv python=3.12.7
```
## Edit the Script
To make sure that everything runs smoothly open the batch file 'runSLC.bat' for editing. This can be done with a rightclick->edit->open in notepad. Make sure the path variables used by the script have the same name as the ones setup in the previous step. Also tweak it in case you are using a different name for the virtual environment or an entire different solution for executing python.
## Usage
Once everything is setup correctly open the provided world file in webots and have a cmd running. Navigate to the folder of 'runSLC.bat' in the cmd. The command:
```
runSLC.bat
```
will activate the python environment, navigate to the webots controller launcher, execute the launcher with the projects main control script, and clean up after execution finishes.

If everthing works fine, you will see in the console output of webots the lines:
>INFO: 'Crazyflie' extern controller: Waiting for local or remote connection on port 1234 targeting robot named 'Crazyflie'.<br>
>INFO: 'Crazyflie' extern controller: connected.<br>
>INFO: '<extern>' controller exited successfully.<br>

while the cmd output reads as 

>"Running the Crazyflie controller"<br>
><br>
>The started controller targets a local instance (ipc protocol) of Webots with port number 1234.<br>
>Targeting the only robot waiting for an extern controller.<br>
>PIDVelocityController initialized<br>
>DroneController initialized<br>
>Distance matrix initialized<br>
>Starting Simulation<br>

The drone should start moving and after a predefined amount of time (set in the main loop of the SelfLocalisationCF.py script) stop as the results are being processed.
Two figures can be found showing the network activity and the position prediction of the network under the folder 'Results'

### Tip
Webots has a control sequence in its taskbar showing the current simulated time and time controls next to it. To speed the execution of the program it is advised to use the accelerated option (Keyborad shortcut: Ctrl+3)



