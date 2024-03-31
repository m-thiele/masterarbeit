# Masterarbeit
This repository provides code for a robust error estimator for the Cahn-Hilliard-Willmore model, which was used in the Master thesis of Michael Thiele. 

## Installation
For the program to run we need fenicsx, which can be installed via conda.
Use the following code to create a new environment with fenicsx.
```
conda env create --name fenics_env --file=requirements.yml
```
Then activate the environment by using
```
conda activate fenicsx_env
```
and install 
```
pip install imageio
pip install imageio-ffmpeg
```


## Running

| filename |  description |
|--------------|-----------|
| cahn_hilliard_willmore.py | computes the solution/residual/eigenvalue   |
| eigenvalue.py |  computes an eigenvalue |
| comp_eps_range.py |  computes the eigenvalue of solutions for different epsilons and plots them |
| compare_residum.py |  computes the residum of solutions for different mesh sizes and plots them |
| residum.py |  methods for computing residuals|


