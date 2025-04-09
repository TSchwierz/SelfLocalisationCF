import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_path_3d(y, y_pred, mse_mean, ID=0):
    stop = int(len(y)/5)
    stop = np.clip(stop, 10, 10000)
    # Unpack the actual and predicted coordinates
    x, y_actual, z = y[:, 0], y[:, 1], y[:, 2]
    x_pred, y_pred_val, z_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2]

    # Create a new figure and 3D axis
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the actual and predicted paths
    ax.plot(x[:stop], y_actual[:stop], z[:stop], label='Actual path', color='blue')
    ax.plot(x_pred[:stop], y_pred_val[:stop], z_pred[:stop], 'r,', label='Predicted path')
    
    # Label axes, legend, and set title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f'First {stop} iterations. Mean MSE = {mse_mean}')

    plt.tight_layout()
    # Save and show the figure
    plt.savefig(f'Results\\ID{ID}\\prediction.png', format='png')
    plt.show()
    plt.close()

def prediction_path_projected(path, path_pred, mse, ID=0):
    x, y, z = path[:,0], path[:,1], path[:,2] 
    x_pred, y_pred, z_pred = path_pred[:,0], path_pred[:,1], path_pred[:,2] 

    ax = plt.figure().add_subplot(projection='3d')

    # Plot x and y axes on z.
    ax.plot(x, y, zs=0, zdir='z', label='curve in (x, y)')
    ax.plot(x, z, zs=0, zdir='y', label='curve in (x, z)')
    ax.plot(y, z, zs=0, zdir='x', label='curve in (y, z)')

    # Make legend, set axes limits and labels
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Customize the view angle so it's easier to see that the scatter points lie
    # on the plane y=0
    #ax.view_init(elev=20., azim=-35, roll=0)
    plt.tight_layout()
    # Save and show the figure
    plt.savefig(f'Results\\ID{ID}\\predictionProjected.png', format='png')
    plt.show()
    plt.close()

def plot_frame_figure(gains, positions_array, network_activity, num_bins=50, neuron=42, ID=0, subID=None):
    """
    Plots a heatmap of network activity at different gain levels and overlays the trajectory. 
    The plot is saved in the results folder within the relative directory.

    :param positions_array: list of shape (ntime, ndim) 
    :param num_bins: int 
    :param network_activity: list of shape (ntime, ngain, nneuron)
    :param neuron: int. Default=42
    """
    x_min, y_min = np.min(positions_array, axis=0)
    x_max, y_max = np.max(positions_array, axis=0)

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, len(gains)+1) 

    # Adding subplots to the gridspec
    for a, alpha in enumerate(gains):

        heatmap_ax = fig.add_subplot(gs[0, a])
        heatmap_ax.set_aspect('equal')

        # Initialize an empty heatmap
        x_bins = np.linspace(x_min, x_max,num_bins)
        y_bins = np.linspace(y_min, y_max,num_bins)
        heatmap = np.zeros((num_bins, num_bins))

        # Iterate over positions and network_activity (Over time)
        for position, activity in zip(positions_array, network_activity):
            x_index = np.digitize(position[0], x_bins) - 1
            y_index = np.digitize(position[1], y_bins) - 1
            heatmap[x_index, y_index] = max(heatmap[x_index, y_index], np.mean(activity[a, neuron]))
            #                                 Activity is of shape (ngains, neurons) here  ^^ pick any neuron

        im = heatmap_ax.imshow(heatmap.T, origin='lower', extent=[x_min, x_max, y_min, y_max], vmax=1, vmin=0)
        heatmap_ax.set(title=f'Gain = {round(alpha, 2)}', xticks=[], yticks=[])
        # add labels left plot
        if a == 0:
            heatmap_ax.set_xlabel('X axis arena')
            heatmap_ax.set_ylabel('Y axis arena')

    # # add subplot for colorbar (there is sth odd here, max fr is a bit above 1 and not 1)
    cbar_ax = fig.add_subplot(gs[0, -1])  # Spanning the last column
    colorbar = fig.colorbar(im, cax=cbar_ax)
    colorbar.set_label('Normalized firing rate', labelpad=15, rotation=270)
    colorbar.set_ticks([0, 0.5, 1])  # Set ticks at min, mid, and max values
    colorbar.set_ticklabels([f'{0:.2f}', f'{0.5:.2f}', f'{1:.2f}'])  # Set tick labels

    positions_array = np.array(positions_array) # enable numpy slicing      
    trajectory_ax = fig.add_subplot(gs[1, 1:len(gains)])  # Adding subplot for the bottom row
    trajectory_ax.plot(positions_array[:, 0], positions_array[:, 1], alpha=0.7, color='purple')
    trajectory_ax.set_title('Effective Arena with travelled path', fontsize=20)
    trajectory_ax.set_aspect('equal')


    fig.tight_layout(h_pad=3.0) # Adjust layout # change spacing between plots
    if subID is None:
        path = f'Results\\ID{ID}\\result_activity_figure.png'
    else:
        path = f'Results\\ID{ID}\\activity_{subID}.png'
    plt.savefig(path, format='png') # save in relative folder Results in Source/Repos/SelfLocalisationCF
    plt.show()
    plt.close()

def plot_modular_activity(prj_op, pos, ac, gains, ID):
    pos2d = np.einsum('mnx, tx->mtn', prj_op, pos)
    for i, p in enumerate(pos2d):
        plot_frame_figure(gains, p, ac[:,i], ID=ID, subID=f'mod{i}')

def plot_3d_trajectory(pos, boundaries, ID='null'):
    x, y, z = pos[:,0], pos[:,1], pos[:,2]

    start, stop = 0, len(x)
    time = np.linspace(start, stop, stop) / stop
    plt.plot(time, x[start:stop], 'b:', label='x')
    plt.plot(time, y[start:stop], 'r:', label='y')
    plt.plot(time, z[start:stop], 'g:', label='z')
    plt.axhline(boundaries[0,0], c='c', label = 'x-y limit')
    plt.axhline(boundaries[0,1], c='c')
    plt.axhline(boundaries[2,0], c='g', label = 'alt limit')
    plt.axhline(boundaries[2,1], c='g')
    plt.ylim(-3, 3.5)
    plt.grid()
    plt.legend()
    print('Plot 1 of 2')
    plt.savefig(f'Results\\ID{ID}\\3d_trajectory_timeplot.png', format='png')
    plt.show()
    plt.close()

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x[:stop], y[:stop], z[:stop], label='path')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    print('Plot 2 of 2')
    plt.show()
    plt.savefig(f'Results\\ID{ID}\\3d_trajectory_spaceplot.png', format='png')
    plt.close()

def plot_fitting_results(n, spacing, score):
    '''
    Plot the scoring of gain configurations as a heatmap based on the number of gains and the spacing between them
    
    :param n: list with the amount of gains
    :param spacing: list with the spacings between gains
    :param score: a flattened list of size (n, spacing) that contains the score of each configuration 
    '''
    heatmap = np.array(score).reshape((len(n), len(spacing)))
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap)
    ax.set_yticks(range(len(n)), labels=n,
                  rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xticks(range(len(spacing)), labels=spacing)

    for i in range(len(n)):
        for j in range(len(spacing)):
            text = ax.text(j, i, f'ID{i*j+j}\n{heatmap[i, j]}',
                           ha="center", va="center", color="w")

    ax.set_title("MSE Scoring over number (y) and spacing (x) of gains")
    fig.colorbar(im)
    fig.tight_layout()
    plt.show()
    plt.savefig(f'Results\\best_gain_results.png', format='png') # save in relative folder Results in Source/Repos/SelfLocalisationCF
    plt.close()