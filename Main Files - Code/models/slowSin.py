from __future__ import print_function
import os
import neat
import visualize
import numpy as np
import matplotlib.pyplot as plt
import math
import pickle


def slowsin(begin, end, numSteps):
    global lstinp
    global lstout
    global x1
    #uterm = 0
    #vterm = 0
    x1 = np.zeros(numSteps)
    x1[0] = 0.0
    x1[1] = (2 * math.pi)/(numSteps)
    lstinp = []
    lstout = []
    t = np.linspace(begin, end, numSteps)
    alpha = (2 * math.pi)/(numSteps)
    for i in range(0,len(t)-2):
        x1[i+2] = ((2 - (alpha ** 2)) * (x1[i+1])) - x1[i]

    for i in range(numSteps - 2):
        tupinp = ()
        tupout = ()
        tuplst = list(tupinp)
        tuplst2 = list(tupout)
        tuplst.append(x1[i])
        tuplst.append(x1[i+1])
        tuplst2.append(x1[i + 2])
        tupinp = tuple(tuplst)
        tupout = tuple(tuplst2)
        lstinp.append(tupinp)
        lstout.append(tupout)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = 100.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        cnt = 0
        lastOutput = 0
        newXi = ()
        totalerror = 0
        for xi, xo in zip(lstinp, lstout):
            if (cnt != 0):
                l = list(newXi)
                l2 = list(xi)
                l.pop(0)
                l.append(lastOutput)
                newXi = tuple(l)
            else:
                l = list(xi)
                newXi = tuple(l)
                cnt = cnt + 1
            output = net.activate(newXi)
            lastOutput = (output[0]) # t[cnt + 1] - t[cnt]
            totalerror += (lastOutput - xo[0]) ** 2
        totalerror = totalerror/(100) # num of time steps
        totalerror = math.sqrt(totalerror)
        if (math.isnan(totalerror) or math.isinf(totalerror)):
            totalerror = 1000000000 # large placeholder number to drastically decrease fitness function
        nodes, connections = size(genome)
        genome.fitness = genome.fitness - totalerror - (0.1 * (nodes + connections))

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         'slowSin-config-feedforward')
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))

    winner = p.run(eval_genomes, 1500)


    with open('finalSlowSinV2', 'wb') as f:
        pickle.dump(winner, f)

    print('\nBest genome:\n{!s}'.format(winner))
    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    global actualout
    global simout
    actualout = []
    simout = []
    pastVal = 0
    cnt1 = 0
    newXi = ()
    for xi, xo in zip(lstinp, lstout):
        if (cnt1 != 0):
            l = list(newXi)
            l2 = list(xi)
            l.pop(0)
            l.append(pastVal)
            newXi = tuple(l)
        else:
            l = list(xi)
            newXi = tuple(l)
            cnt1 = cnt1 + 1
        output = winner_net.activate(newXi)
        pastVal = (output[0])
        actualout.append(xo[0])
        simout.append(pastVal)

    node_names = {-1: 'x[t]', -2: 'x[t+1]', 0: 'x[t+2]'}
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)
    visualize.draw_net(config, winner, view=True, node_names=node_names, filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)

    return actualout, simout
def slowSinTest():
    with open('finalSlowSin', 'rb') as f:
        c = pickle.load(f)

    print('Loaded genome:')
    print(c)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'slowSin-config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    winner_net = neat.nn.FeedForwardNetwork.create(c, config)
    slowsin(0, 200, 200)
    actualout = []
    simout = []
    totalerror = 0
    cnt1 = 0
    pastVal = 1
    newXi = ()
    for xi, xo in zip(lstinp, lstout):
        if (cnt1 != 0):
            l = list(newXi)
            l2 = list(xi)
            l.pop(0)
            l.append(pastVal)
            newXi = tuple(l)
        else:
            l = list(xi)
            newXi = tuple(l)
            cnt1 = cnt1 + 1
        output = winner_net.activate(newXi)
        pastVal = (output[0])
        actualout.append(xo[0])
        simout.append(pastVal)
    return actualout, simout
