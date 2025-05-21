@echo off

REM Activate the conda environment
call conda activate myenv

cd %SLcontroller%\
python analysis.py $1

REM Change back to the specified home directory
cd %specHome%

REM Deactivate the conda environment
call conda deactivate

@echo on