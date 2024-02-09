import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
mpl.rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r"\usepackage{amsmath} \usepackage{bold-extra}"
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from matplotlib.ticker import FormatStrFormatter

# for line graphs
def lineGraph(t, y1, y2, boundaryX, label='', xlabel='', ylabel='', y1label = '', y2label = '', ylimlow='', ylimhigh='', save_path=None):
    plt.figure(figsize=(4.5, 4.5)) # r'\textsc{base}'
    plt.plot(t, y1, 'r-')
    plt.plot(t, y2, 'b-', linestyle='dashed')
    plt.ylabel(ylabel, fontsize=10) # 'Precison$_{PV=a}$/Precision$_{PV=b}$'
    plt.xlabel(xlabel, fontsize=10) # '$|1-r|$'
    #plt.axhline(1, linestyle='dashed')
    #plt.axvline(0, linestyle='dashed')
    plt.ylim(ylimlow, ylimhigh)
    #plt.xticks(fontsize=12)
    #plt.xticks(fontsize=12)
    plt.legend([y1label, y2label])  # ncol=2
    plt.axvline(x = boundaryX, color = 'k')
    plt.axvspan(0, boundaryX, color = 'lightgray')
    plt.title(label)
    if save_path:
        print(save_path)
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()

# graphs performance given x1 in dataset size and y1 in running time
def graphPerf(x1, y1, label = '', xlabel='', ylabel='', save_path=None):
    plt.figure(figsize=(4.5, 4.5)) # r'\textsc{base}'
    plt.scatter(x1, y1, label=label, marker='o',s=50,c='steelblue')
    plt.ylabel(ylabel, fontsize=10) # 'Precison$_{PV=a}$/Precision$_{PV=b}$'
    plt.xlabel(xlabel, fontsize=10) # '$|1-r|$'
    plt.title(label)
    if save_path:
        print(save_path)
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
