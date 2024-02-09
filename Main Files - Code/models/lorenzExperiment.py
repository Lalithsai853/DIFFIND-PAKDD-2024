from __future__ import print_function
import os
import neat
import visualize
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle


def lorenzN(begin, end, numSteps, omega, beta, rho):
    global lstinp
    global lstout
    global x1
    global y1
    global z1
    x1 = np.ones(numSteps)
    y1 = np.ones(numSteps)
    z1 = np.ones(numSteps)
    lstinp = []
    lstout = []
    x1[0] = 1
    y1[0] = 1
    z1[0] = 1
    t = np.linspace(begin, end, numSteps)
    print(begin)
    print(end)
    print(numSteps)
    for i in range(0,len(t)-1):
        dxdt = omega * (y1[i] - x1[i])
        dydt = (x1[i] * (rho - z1[i])) - y1[i]
        dzdt = (x1[i] * y1[i]) - (beta * z1[i])
        x1[i+1] = x1[i] + ((dxdt) * (t[i+1] - t[i]))
        y1[i+1] = y1[i] + ((dydt) * (t[i+1] - t[i]))
        z1[i+1] = z1[i] + ((dzdt) * (t[i+1] - t[i]))
    for i in range(numSteps - 1):
        tupinp = ()
        tupout = ()
        tuplst = list(tupinp)
        tuplst2 = list(tupout)
        tuplst.append(x1[i])
        tuplst.append(y1[i])
        tuplst.append(z1[i])
        tuplst2.append(x1[i + 1])
        tupinp = tuple(tuplst)
        tupout = tuple(tuplst2)
        lstinp.append(tupinp)
        lstout.append(tupout)

def lorenzTemp(begin, end, numSteps, numLag, omega, beta, rho):
    x1 = np.ones(numSteps)
    y1 = np.ones(numSteps)
    z1 = np.ones(numSteps)
    t = np.linspace(begin, end, numSteps)
    finallstinpX = []
    finallstoutX = []
    finallstinpY = []
    finallstoutY = []
    finallstinpZ = []
    finallstoutZ = []
    for i in range(0,len(t)-1):
        dxdt = omega * (y1[i] - x1[i])
        dydt = (x1[i] * (rho - z1[i])) - y1[i]
        dzdt = (x1[i] * y1[i]) - (beta * z1[i])
        x1[i+1] = x1[i] + ((dxdt) * (t[i+1] - t[i]))
        y1[i+1] = y1[i] + ((dydt) * (t[i+1] - t[i]))
        z1[i+1] = z1[i] + ((dzdt) * (t[i+1] - t[i]))
    for i in range(numSteps - 1):
        tupinpX = ()
        tupoutX = ()
        tupinpY = ()
        tupoutY = ()
        tupinpZ = ()
        tupoutZ = ()
        lstinpX = list(tupinpX)
        lstoutX = list(tupoutX)
        lstinpY = list(tupinpY)
        lstoutY = list(tupoutY)
        lstinpZ = list(tupinpZ)
        lstoutZ = list(tupoutZ)
        lstinpX.append(x1[i])
        lstinpX.append(y1[i])
        lstinpX.append(z1[i])
        lstoutX.append(x1[i + 1])
        lstinpY.append(x1[i])
        lstinpY.append(y1[i])
        lstinpY.append(z1[i])
        lstoutY.append(y1[i + 1])
        lstinpZ.append(x1[i])
        lstinpZ.append(y1[i])
        lstinpZ.append(z1[i])
        lstoutZ.append(z1[i + 1])
        tupinpX = tuple(lstinpX)
        tupoutX = tuple(lstoutX)
        tupinpY = tuple(lstinpY)
        tupoutY = tuple(lstoutY)
        tupinpZ = tuple(lstinpZ)
        tupoutZ = tuple(lstoutZ)
        finallstinpX.append(tupinpX)
        finallstoutX.append(tupoutX)
        finallstinpY.append(tupinpY)
        finallstoutY.append(tupoutY)
        finallstinpZ.append(tupinpZ)
        finallstoutZ.append(tupoutZ)

    return finallstinpX, finallstoutX, finallstinpY, finallstoutY, finallstinpZ, finallstoutZ

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 100.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        cnt = 0
        lastOutput = 1
        newXi = ()
        totalerror = 0
        for xi, xo in zip(lstinp, lstout):
            if (cnt != 0):
                l = list(newXi)
                l2 = list(xi)
                l.pop(0)
                l.pop(0)
                l.pop(0)
                l.append(lastOutput)
                l.append(l2[1])
                l.append(l2[2])
                newXi = tuple(l)
            else:
                l = list(xi)
                newXi = tuple(l)
                cnt = cnt + 1
            output = net.activate(newXi)
            lastOutput += ((output[0]) * (10/20000))
            totalerror += (lastOutput - xo[0]) ** 2
        totalerror = math.sqrt(totalerror/(20000))
        if (math.isnan(totalerror) or math.isinf(totalerror)):
            totalerror = 1000000
        nodes, connections = size(genome)
        genome.fitness = genome.fitness - totalerror - (0.1 * (nodes + connections))

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'lorenz-config-feedforward')
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 100)

    with open('finalLorenzXV2', 'wb') as f:
        pickle.dump(winner, f)

    print('\nBest genome:\n{!s}'.format(winner))
    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    global actualout
    global simout
    actualout = []
    simout = []
    pastVal = 1
    cnt1 = 0
    newXi = ()
    for xi, xo in zip(lstinp, lstout):
        if (cnt1 != 0):
            l = list(newXi)
            l2 = list(xi)
            l.pop(0)
            l.pop(0)
            l.pop(0)
            l.append(pastVal)
            l.append(l2[1])
            l.append(l2[2])
            newXi = tuple(l)
        else:
            l = list(xi)
            newXi = tuple(l)
            cnt1 = cnt1 + 1
        output = winner_net.activate(newXi)
        actualout.append(xo[0])
        simout.append(pastVal + (output[0] * (10/20000)))
        pastVal += (output[0] * (10/20000))

    node_names = {-1: 'x[t-1]', -2: 'y[t-1]', -3: 'z[t-1]', 0: 'dx/dt'}
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    visualize.draw_net(config, winner, view=True, node_names=node_names, filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)


    return actualout, simout


def lorenzTest():
    with open('finalLorenzXV2', 'rb') as f:
        c = pickle.load(f)

    with open('finalLorenzYV2', 'rb') as f:
        d = pickle.load(f)

    with open('finalLorenzZV2', 'rb') as f:
        e = pickle.load(f)

    print('Loaded genomes:')
    print(c)
    print(d)
    print(e)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'lorenz-config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    winner_netX = neat.nn.FeedForwardNetwork.create(c, config)
    winner_netY = neat.nn.FeedForwardNetwork.create(d, config)
    winner_netZ = neat.nn.FeedForwardNetwork.create(e, config)
    inpX, outX, inpY, outY, inpZ, outZ = lorenzTemp(0, 20, 40000, 5, 10, 2.6667, 28)
    simoutX = []
    simoutY = []
    simoutZ = []
    actualoutX = []
    actualoutY = []
    actualoutZ = []
    totalerror = 0
    cnt1 = 0
    pastValX = 1
    pastValY = 1
    pastValZ = 1
    newXi = ()
    for xi, xo in zip(inpX, outX):
        if (cnt1 != 0):
            l = list(newXi)
            l2 = list(xi)
            l.pop(0)
            l.pop(0)
            l.pop(0)
            l.append(pastValX)
            l.append(pastValY)
            l.append(pastValZ)
            newXi = tuple(l)
        else:
            l = list(xi)
            newXi = tuple(l)
        outputX = winner_netX.activate(newXi)
        outputY = winner_netY.activate(newXi)
        outputZ = winner_netZ.activate(newXi)
        actualoutX.append(xo[0])
        actualoutY.append(outY[cnt1][0])
        actualoutZ.append(outZ[cnt1][0])
        simoutX.append(pastValX + (outputX[0] * (20/40000)))
        simoutY.append(pastValY + (outputY[0] * (20/40000)))
        simoutZ.append(pastValZ + (outputZ[0] * (20/40000)))
        pastValX += (outputX[0] * (20/40000))
        pastValY += (outputY[0] * (20/40000))
        pastValZ += (outputZ[0] * (20/40000))
        cnt1 += 1
    return actualoutX, actualoutY, actualoutZ, simoutX, simoutY, simoutZ
