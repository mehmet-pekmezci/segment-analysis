import pickle

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    file = open("../input/synthetic_dataset.dat", 'rb')
    data_set = pickle.load(file)[0]
    file.close()
    first_segment_length_array = [5, 10, 15, 20, 25, 30, 35, 40, 45]

    fig, axs= plt.subplots(3, 3, figsize=(18, 18))

    for i, first_segment_length in enumerate(first_segment_length_array):
        row = int(i / 3)
        col = i % 3
        ax = axs[row][col]
        for traj in data_set:
            x = np.arange(traj.shape[0])
            y = traj[:, 1]
            ax.scatter(x[0:first_segment_length], y[0:first_segment_length], color='blue', label='First Segment')
            ax.scatter(x[first_segment_length:], y[first_segment_length:], color='orange', label='Second Segment')
            ax.tick_params(axis='y', labelsize=12)
            ax.tick_params(axis='x', labelsize=12)
            ax.axvline(x=first_segment_length, color='red', linestyle='--', linewidth=5, label='Segment Border')
            if col == 0:
                ax.set_ylabel('Position', fontsize=15)
            if row == 2:
                ax.set_xlabel('Timestep', fontsize=15)
        ax.set_title(f"Segment Border at {first_segment_length}", fontsize=25)
    plt.suptitle("Analyzed Segment Borders", fontsize=50)
    plt.savefig('/Users/mehmetpekmezci/Desktop/test_plot.png')
    plt.close(fig)

if __name__ == '__main_1_':
    plt.plot(np.arange(10), np.arange(10)/10)
    plt.show()