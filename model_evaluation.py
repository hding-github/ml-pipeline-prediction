

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def calculate_r2(tD_results):
    r2 = r2_score(tD_results["y_test"], tD_results["y_pred"])
    return r2

def calculate_mse(y_true, y_pred):
    # Mean Squared Error 
    tMSE = np.square(np.subtract(y_true, y_pred)).mean() 
    return tMSE

def plot_model_performance_3d(tD_Data):
    strPath = "./results/"
    strFile = "plot_performance_at_different_settings.png" 
    strPathFile = strPath + strFile
    #To observe model performance 

    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    import numpy as np

    #fig = plt.figure()
    fig = plt.figure(figsize=(12, 10))
    #ax.view_init(elev=30, azim=45, roll=15)
    ax = fig.add_subplot(111, projection = "3d")

    ax.set_xlabel(tD_Data["xlabel"])
    ax.set_ylabel(tD_Data["ylabel"]) 
    ax.set_zlabel(tD_Data["zlabel"]) 
    ax.set_title(tD_Data["title"])

    #ax.set_xlim3d(0,10)
    #ax.set_ylim3d(0,10) 

    #xpos = [2,5,8,2,5,8,2,5,8]
    #ypos = [1,1,1,5,5,5,9,9,9]
    #zpos = np.zeros(9)

    #dx = np.ones(9)
    #dy = np.ones(9)
    #dz = [np.random.random(9) for i in range(4)]  # the heights of the 4 bar sets

    xpos = tD_Data["xpos"]
    ypos = tD_Data["ypos"]
    zpos = np.ones_like(xpos)*0.0

    dx = np.ones_like(xpos)*0.25
    dy = np.ones_like(ypos)*0.25
    dz = tD_Data["dz"]

    tNumOfSegments = len(dz)

    _zpos = zpos   # the starting zpos for each bar
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    for i in range(tNumOfSegments):
        ax.bar3d(xpos, ypos, _zpos, dx, dy, dz[i], color=colors[i])
        _zpos += dz[i]    # add the height of each bar to know where to start the next

    #plt.gca().invert_xaxis()
    #plt.show()
    plt.savefig(strPathFile)

def plot_scatter_true_pred(y_true, y_pred):
    strPath = "./results/"
    strFile = "plot_y_true_pred.png" 
    strPathFile = strPath + strFile
    plt.figure(figsize=(12, 10))
    plt.scatter(y_true, y_pred, marker='*')

    # naming the x axis
    plt.xlabel('Training data: y - true')
    # naming the y axis
    plt.ylabel('Training data: y - pred')

    # giving a title to my graph
    plt.title('Scatter Plot of Prediction Outcomes')

    plt.savefig(strPathFile)

def plot_demo():
    strPath = "./results/"
    strFile = "plot_performance_at_different_settings.png" 
    strPathFile = strPath + strFile
    plt.figure(figsize=(24, 20))
    plt.style.use('_mpl-gallery')

    # Make data
    x = [1, 1, 2, 2]
    y = [1, 2, 1, 2]
    z = [0, 0, 0, 0]
    dx = np.ones_like(x)*0.5
    dy = np.ones_like(x)*0.5
    dz = [2, 3, 1, 4]

    # Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.bar3d(x, y, z, dx, dy, dz)

    ax.set(xticklabels=[],
        yticklabels=[],
        zticklabels=[])

    plt.savefig(strPathFile)

#plot_demo()