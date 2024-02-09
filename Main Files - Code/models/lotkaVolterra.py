from __future__ import print_function
import os
import neat
import visualize
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle


def lotkaVolterra(begin, end, numSteps, numLag, alpha, beta, delta, gamma):
    global lstinp
    global lstout
    global x1
    global y1
    x1 = np.ones(numSteps)
    y1 = np.ones(numSteps)
    x1[0] = 3.0
    y1[0] = 1.0
    lstinp = []
    lstout = []
    t = np.linspace(begin, end, numSteps)
    for i in range(0,len(t)-1):
        dxdt = ((alpha) * (x1[i])) - (beta * x1[i] * y1[i])
        dydt = (delta * x1[i] * y1[i]) - (gamma * y1[i])
        x1[i+1] = x1[i] + ((dxdt) * (t[i+1] - t[i]))
        y1[i+1] = y1[i] + ((dydt) * (t[i+1] - t[i]))

    for i in range(numSteps - 1):
        tupinp = ()
        tupout = ()
        tuplst = list(tupinp)
        tuplst2 = list(tupout)
        tuplst.append(x1[i])
        tuplst.append(y1[i])
        tuplst2.append(y1[i + 1])
        tupinp = tuple(tuplst)
        tupout = tuple(tuplst2)
        lstinp.append(tupinp)
        lstout.append(tupout)

def lotkaVolterraTemp(begin, end, numSteps, alpha, beta, delta, gamma):
    x1 = np.ones(numSteps)
    y1 = np.ones(numSteps)
    t = np.linspace(begin, end, numSteps)
    x1[0] = 3.0
    y1[0] = 1.0
    finallstinpX = []
    finallstoutX = []
    finallstinpY = []
    finallstoutY = []
    for i in range(0,len(t)-1):
        dxdt = ((alpha) * (x1[i])) - (beta * x1[i] * y1[i])
        dydt = (delta * x1[i] * y1[i]) - (gamma * y1[i])
        x1[i+1] = x1[i] + ((dxdt) * (t[i+1] - t[i]))
        y1[i+1] = y1[i] + ((dydt) * (t[i+1] - t[i]))
    for i in range(numSteps - 1):
        tupinpX = ()
        tupoutX = ()
        tupinpY = ()
        tupoutY = ()
        lstinpX = list(tupinpX)
        lstoutX = list(tupoutX)
        lstinpY = list(tupinpY)
        lstoutY = list(tupoutY)
        lstinpX.append(x1[i])
        lstinpX.append(y1[i])
        lstoutX.append(x1[i + 1])
        lstinpY.append(x1[i])
        lstinpY.append(y1[i])
        lstoutY.append(y1[i + 1])
        tupinpX = tuple(lstinpX)
        tupoutX = tuple(lstoutX)
        tupinpY = tuple(lstinpY)
        tupoutY = tuple(lstoutY)
        finallstinpX.append(tupinpX)
        finallstoutX.append(tupoutX)
        finallstinpY.append(tupinpY)
        finallstoutY.append(tupoutY)
    return finallstinpX, finallstoutX, finallstinpY, finallstoutY

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 100.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        cnt = 0
        lastOutput = 1.0
        newXi = ()
        totalerror = 0
        for xi, xo in zip(lstinp, lstout):
            if (cnt != 0):
                l = list(newXi)
                l2 = list(xi)
                l.pop(0)
                l.pop(0)
                l.append(l2[0])
                l.append(lastOutput)
                newXi = tuple(l)
            else:
                l = list(xi)
                newXi = tuple(l)
                cnt = cnt + 1
            output = net.activate(newXi)
            lastOutput += ((output[0]) * (10/20000)) # change values for (length of timseries)/(num of steps) depending on experiment
            totalerror += (lastOutput - xo[0]) ** 2
        totalerror = totalerror/(20000)
        totalerror = math.sqrt(totalerror)
        if (math.isnan(totalerror) or math.isinf(totalerror)):
            totalerror = 100000000
        nodes, connections = size(genome)
        genome.fitness = genome.fitness - totalerror - (0.1 * (nodes + connections))

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'lotkaVolterra-config-feedforward')
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 100)

    with open('finalLotkaVolterraX', 'wb') as f:
        pickle.dump(winner, f)

    print('\nBest genome:\n{!s}'.format(winner))
    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    global actualout
    global simout
    actualout = []
    simout = []
    pastVal = 1.0
    cnt1 = 0
    newXi = ()
    for xi, xo in zip(lstinp, lstout):
        if (cnt1 != 0):
            l = list(newXi)
            l2 = list(xi)
            l.pop(0)
            l.pop(0)
            l.append(l2[0])
            l.append(pastVal)
            newXi = tuple(l)
        else:
            l = list(xi)

            newXi = tuple(l)
            cnt1 = cnt1 + 1
        output = winner_net.activate(newXi)
        actualout.append(xo[0])
        simout.append(pastVal + (output[0] * (10/20000)))
        pastVal += (output[0] * (10/20000))


    node_names = {-1: 'x[t-1]', -2: 'y[t-1]', 0: 'dx/dt'}
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    visualize.draw_net(config, winner, view=True, node_names=node_names, filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)

    return actualout, simout

def lotkaVolterraTest():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'lotkaVolterra-config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    with open('finalLotkaVolterraX', 'rb') as f:
        c = pickle.load(f)

    with open('finalLotkaVolterraY', 'rb') as f:
        d = pickle.load(f)

    print('Loaded genomes:')
    print(c)
    print(d)
    winner_netX = neat.nn.FeedForwardNetwork.create(c, config)
    winner_netY = neat.nn.FeedForwardNetwork.create(d, config)
    inpX, outX, inpY, outY = lotkaVolterraTemp(0, 20, 40000, 1.1, 0.4, 0.1, 0.4)
    simoutX = []
    simoutY = []
    actualoutX = []
    actualoutY = []
    totalerror = 0
    cnt1 = 0
    pastValX = 3.0
    pastValY = 1.0
    newXi = ()
    for xi, xo in zip(inpX, outX):
        if (cnt1 != 0):
            l = list(newXi)
            l2 = list(xi)
            l.pop(0)
            l.pop(0)
            l.append(pastValY)
            l.append(pastValY)
            newXi = tuple(l)
        else:
            l = list(xi)
            newXi = tuple(l)
        outputX = winner_netX.activate(newXi)
        outputY = winner_netY.activate(newXi)
        actualoutX.append(xo[0])
        actualoutY.append(outY[cnt1][0])
        simoutX.append(pastValX + (outputX[0] * (20/40000)))
        simoutY.append(pastValY + (outputY[0] * (20/40000)))
        pastValX += (outputX[0] * (20/40000))
        pastValY += (outputY[0] * (20/40000))
        cnt1 += 1
    return actualoutX, actualoutY, simoutX, simoutY
