"""
Various utility routines for making plots look prettier
"""

import numpy as np
from math import atan2,degrees
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText

## Label line with line2D label data
## Reference
## https://stackoverflow.com/questions/16992038/inline-labels-in-matplotlib
def labelLine(line,x,label=None,align=True,**kwargs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        dx = xdata[ip] - xdata[ip-1]
        dy = ydata[ip] - ydata[ip-1]
        ang = degrees(atan2(dy,dx))

        #Transform to screen co-ordinates
        pt = np.array([x,y]).reshape((1,2))
        trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]

    else:
        trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    ax.text(x,y,label,rotation=trans_angle,**kwargs)

def labelLines(lines,align=True,xvals=None,**kwargs):

    ax = lines[0].axes
    labLines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            labLines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        xvals = np.linspace(xmin,xmax,len(labLines)+2)[1:-1]

    for line,x,label in zip(labLines,xvals,labels):
        labelLine(line,x,label,align,**kwargs)


## How to make a legend with only text
## https://stackoverflow.com/questions/31661576/how-to-make-a-legend-with-only-text
def textonly(ax, txt, fontsize=14, loc=2, *args, **kwargs):

    at = AnchoredText(txt, prop=dict(size=fontsize), frameon=True, loc=loc)
    at.patch.set_boxstyle("round, pad=0., rounding_size=0.2")
    ax.add_artist(at)

    return at

if __name__ == '__main__':
    
    from matplotlib import pyplot as plt
    from scipy.stats import loglaplace,chi2

    X = np.linspace(0, 1, 500)
    A = [1, 2, 5, 10, 20]
    funcs = [np.arctan, np.sin, loglaplace(4).pdf, chi2(5).pdf]

    plt.subplot(221)
    for a in A:
        plt.plot(X, np.arctan(a*X), label=str(a))

    labelLines(plt.gca().get_lines(), zorder=2.5)

    plt.subplot(222)
    for a in A:
        plt.plot(X, np.sin(a*X), label=str(a))

    labelLines(plt.gca().get_lines(), align=False, fontsize=14)

    plt.subplot(223)
    for a in A:
        plt.plot(X, loglaplace(4).pdf(a*X), label=str(a))

    xvals = [0.8, 0.55, 0.22, 0.104, 0.045]
    labelLines(plt.gca().get_lines(), align=False, xvals=xvals, color='k')

    plt.subplot(224)
    for a in A:
        plt.plot(X, chi2(5).pdf(a*X), label=str(a))

    lines = plt.gca().get_lines()
    l1=lines[-1]
    labelLine(l1,0.6,label=r'$Re=${}'.format(l1.get_label()),ha='left',va='bottom',align = False)
    labelLines(lines[:-1],align=False)
    
    #plt.show()
        
