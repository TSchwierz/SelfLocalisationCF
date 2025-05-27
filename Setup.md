# Setup
This project and tutorial have been written with a Windows 10 implementation in mind. In order to use this project, an installation of webots and the ability to run Python on your machine are required.

## Webots
Under https://cyberbotics.com you can download the latest release of webots, a simulation environment to work on robotics.

Once it is installed and opened, navigate to File/Open World and select the world file 'SquareBox.wbt' that is part of this repository in the "Webots&Execution" folder. 
## Environmental Path Variables
In order to run the project, the crazyflie drone in webots needs to be targeted by an external controller, a Python script in this case. This can be done by passing the script to the webots-controller executable in the cmd.
```
webots-controller.exe controller.py
```
A tutorial on that can be found [here](https://cyberbotics.com/doc/guide/running-extern-robot-controllers?tab-os=windows). 

To make this easier, a batch script has been written for which it is recommended to create some path variables. In Windows, go to settings and search for 'environmental path variables'. Add the following three variables:
- WEBOTS : pointing towards the folder of 'webots-controller.exe'. Usually this is C:\Program Files\Webots\msys64\mingw64\bin
- SLController : pointing to the local copy of the repository containing the file 'SelfLocalisationCF.py'.
- specHome : This points to the folder from which the batch script 'runSLC.bat' is run.

## Python
In order to execute the code, your machine needs to be able to run Python. In my case, I set up a virtual environment using [anaconda](https://www.anaconda.com) using the following command in the anaconda prompt:
```
conda create --name myenv python=3.12.7
```

Once the environment is create, activate it with
```
conda activate myenv
```
There is a number of packages that need to be installed before the script will be able to run. Activate the environment and download the following python packages:
- numpy
- matplotlib
- scipy
- scikitlearn
- Jupyter
- numba
- (possible non-exhaustive list)

To download a package it is prefered to use
```
conda install [package name]
```
in the activated virtual environment

## Edit the Script
To make sure that everything runs smoothly, open the batch file 'runSLC.bat' for editing. This can be done with a right-click->edit->open in Notepad. Make sure the path variables used by the script have the same name as the ones setup in the previous step. Also tweak it in case you are using a different name for the virtual environment or an entire different solution for executing Python.
## Usage
Once everything is set up correctly, open the provided world file in webots and have a cmd running. Navigate to the folder of 'runSLC.bat' in the cmd. The command:
```
runSLC.bat
```
will activate the Python environment, navigate to the webots controller launcher, execute the launcher with the project's main control script, and clean up after execution finishes.

If everything works fine, you will see in the console output of webots the lines:
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

The drone should start moving and, after a predefined amount of time (set in the main loop of the SelfLocalisationCF.py script) stop as the results are being processed.
All relevant data of the simulation are stored in a dictionary in a pickle file under the folder 'Results'. Pickle is a python library that readily provides (de-)serialisation of data structures.

### Tip
Webots has a control sequence in its taskbar showing the current simulated time and time controls next to it. To speed the execution of the program, it is advised to use the accelerated option (Keyboard shortcut: Ctrl+3)

### Analysis
In the main repository directory is a Jupyter Notebook file called "ResultAnalysis.ipynb". This notebook has code that opens the pickle data file and processes the data, resulting in several plots.