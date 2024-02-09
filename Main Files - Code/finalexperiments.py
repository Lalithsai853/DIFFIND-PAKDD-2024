from __future__ import print_function
import os
import re
import neat
import visualize
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import pickle
import gzip
import matplotlib.pyplot as plt
import time
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath} \usepackage{bold-extra}"
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from matplotlib.ticker import FormatStrFormatter


from lorenzExperiment import run as runLorenz
from lorenzExperiment import *
from slowSin import run as runSlowSin
from slowSin import *
from vanderPol import run as runVanderPol
from vanderPol import *
from lotkaVolterra import run as runLotkaVolterra
from lotkaVolterra import *
from general import *

def lorenzE(path):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'lorenz-config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    lorenzN(0, 10, 20000, 10, (8/3), 28)
    actual, simulated = runLorenz(config_path)
    print(len(actual))
    print(len(simulated))
    startVal = 1
    actual.insert(0, startVal)
    simulated.insert(0, startVal)
    min = 0
    max = 10
    steps = 20000
    t = np.linspace(min, max, steps)
    lineGraph(t, actual, simulated, (max/2), "Lorenz X Variable vs. Time (Training)", "Time", "Output", "Real", "Simulated", path)

    startVal = 1
    actualX, actualY, actualZ, simX, simY, simZ = lorenzTest()
    min = 0
    max = 20
    steps = 40000
    actualX.insert(0, startVal)
    actualY.insert(0, startVal)
    actualZ.insert(0, startVal)
    simX.insert(0, startVal)
    simY.insert(0, startVal)
    simZ.insert(0, startVal)
    t = np.linspace(min, max, steps)
    lineGraph(t, actualX, simX, (max/2), "Lorenz X Variable vs. Time (Test)", "Time", "Output", "Real", "Simulated", path)
    lineGraph(t, actualY, simY, (max/2), "Lorenz Y Variable vs. Time (Test)", "Time", "Output", "Real", "Simulated", path)
    lineGraph(t, actualZ, simZ, (max/2), "Lorenz Z Variable vs. Time (Test)", "Time", "Output", "Real", "Simulated", path)


def slowSinE(path):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,'slowSin-config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    slowsin(0, 100, 100)
    numSteps = 100
    actual, simulated = runSlowSin(config_path)
    print(len(actual))
    print(len(simulated))
    startVal = 0.0
    actual.insert(0, (2 * math.pi)/(numSteps))
    simulated.insert(0, (2 * math.pi)/(numSteps))
    actual.insert(0, startVal)
    simulated.insert(0, startVal)
    min = 0
    max = 100
    steps = 100
    t = np.linspace(min, max, steps)
    lineGraph(t, actual, simulated, (max/2), "Slow Sinusoid vs. Time (Train)", "Time", "Output", "Real", "Simulated", -2, 2, path)

    actual, simulated = slowSinTest()
    min = 0
    max = 200
    steps = 200
    actual.insert(0, (2 * math.pi)/(steps))
    simulated.insert(0, (2 * math.pi)/(steps))
    actual.insert(0, startVal)
    simulated.insert(0, startVal)
    t = np.linspace(min, max, steps)
    lineGraph(t, actual, simulated, "Slow Sinusoid vs. Time (Test)", "Time", "Output", "Real", "Simulated", path)

def vanderPolE(path):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'vanderPol-config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    vanderPol(0, 10, 80000, 3)
    actual, simulated = runVanderPol(config_path)
    print(len(actual))
    print(len(simulated))
    startValX = 2.0
    startValY = 0.0
    actual.insert(0, startValX)
    simulated.insert(0, startValX)
    min = 0
    max = 10
    steps = 80000
    t = np.linspace(min, max, steps)
    lineGraph(t, actual, simulated, (max/2),"Van der Pol X Variable vs. Time (Training)", "Time", "Output", "Real", "Simulated", path)

    startValX = -2.0
    startValY = 0.0
    actualX, actualY, simX, simY = vanderPolTest()
    min = 0
    max = 20
    steps = 40000
    actualX.insert(0, startValX)
    actualY.insert(0, startValY)
    simX.insert(0, startValX)
    simY.insert(0, startValY)
    t = np.linspace(min, max, steps)
    lineGraph(t, actualX, simX, (max/2), "Van der Pol X Variable vs.Time", "Time", "Output", "Real", "Simulated", -4, 4, path)
    lineGraph(t, actualY, simY, (max/2), "Van der Pol Y Variable vs.Time", "Time", "Output", "Real", "Simulated", -2, 2, path)
    actual, simulated = vanderPolTest()
    min = 0
    max = 30
    steps = 60000
    actual.insert(0, startVal)
    simulated.insert(0, startVal)
    t = np.linspace(min, max, steps)
    lineGraph(t, actual, simulated, "Van der Pol Y Variable vs. Time (Test)", "Time", "Output", "Real", "Simulated", path)

def lotkaVolterraE(path):
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'lotkaVolterra-config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    lotkaVolterra(0, 10, 20000, 1.1, 0.4, 0.1, 0.4)
    actual, simulated = runLotkaVolterra(config_path)
    print(len(actual))
    print(len(simulated))
    startValX = 3.0
    startValY = 1.0
    actual.insert(0, startValY)
    simulated.insert(0, startValY)
    min = 0
    max = 10
    steps = 20000
    t = np.linspace(min, max, steps)
    lineGraph(t, actual, simulated, (max/2), "Lotka Volterra X Variable vs. Time (Training)", "Time", "Output", "Real", "Simulated", path)


    startValX = 3.0
    startValY = 1.0
    actualX, actualY, simX, simY = lotkaVolterraTest()
    min = 0
    max = 20
    steps = 40000
    actualX.insert(0, startValX)
    actualY.insert(0, startValY)
    simX.insert(0, startValX)
    simY.insert(0, startValY)
    t = np.linspace(min, max, steps)


    lineGraph(t, actualX, simX, (max/2), "Lotka Volterra X Variable vs.Time", "Time", "Output", "Real", "Simulated", 0, 18, path)
    lineGraph(t, actualY, simY, (max/2), "Lotka Volterra Y Variable vs.Time", "Time", "Output", "Real", "Simulated", 0, 12, path)


    with open('finalLotkaVolterraX', 'rb') as f:
        c = pickle.load(f)
    with open('finalLotkaVolterraY', 'rb') as f:
        d = pickle.load(f)

    print('Loaded genome:')
    print(c)
    print(d)



if __name__ == "__main__":
    # Can uncomment any experiment to run them
    path = "INSERT PATH"
    #lorenzE(path)
    #slowSinE(path)
    #vanderPolE(path)
    #lotkaVolterraE(path)
