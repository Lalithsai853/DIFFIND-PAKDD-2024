# DIFFIND-PAKDD-2024
This is the code repository for DIFFIND. ll the experiments are run on a stock desktop with 3.4 GHz 4-core CPU and 16 GB RAM, running Ubuntu 20.04 LTS.

Necessary packages to run:
- numpy
- matplotlib
- visualize
- pickle
- gzip
- neat-python

Files:
- finalexperiments.py: main file containing all experiments
- general.py: used to graph results for training/testing

Models:
- vanderPol.py: contains methods to run Van Der Pol experiment
- lotkaVolterra.py: contains methods to run Lotka Volterra experiment
- lorenz.py: contains methods to run the Lorenz experiment
- slowSin.py: contains methods to run the slow sinusoid experiment

Config Files:
- lotkaVolterra-config-feedforward: config file with parameters for Lotka Volterra experiment
- vanderPol-config-feedforward: config file with parameters for Van Der Pol experiment
- slowSin-config-feedforward: config file with parameters for slow sinusoid experiment
- lorenz-config-feedforward: config file with parameters for Lorenz experiment
