@echo off

REM Activate the conda environment
call conda activate myenv

echo "Running the Crazyflie controller"

REM Change to the Webots directory
cd %WEBOTS%

REM Execute the controller script with Webots
webots-controller.exe %SLcontroller%\SelfLocalisationCF.py

REM Change back to the specified home directory
cd %specHome%

REM Deactivate the conda environment
call conda deactivate

@echo on