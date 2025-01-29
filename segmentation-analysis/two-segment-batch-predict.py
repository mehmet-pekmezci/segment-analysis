import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from trainer import CCnmpTrainer, CCnmp2
from trajectory_printer import print_cnmp_trajectory

if __name__ == "__main__":
    file = open("../input/synthetic_dataset.dat", 'rb')
    data_set = pickle.load(file)
    file.close()
    trajectories = np.zeros((4, 50))
    for i, trajectory in enumerate(data_set[0]):
        trajectories[i] = trajectory[:, 1]

    trial_id = time.time()
    for iter in range(1):
        y_arr = []
        x_arr = []

        for trajectory in trajectories:
            x_arr.append(np.linspace(0, 1, len(trajectory)).reshape((-1, 1)))
            y_arr.append(trajectory.reshape((-1, 1)))

        data_x = np.array(x_arr)
        data_y = np.array(y_arr)

        error_dict = {}
        base_folder_name = "/Users/mehmetpekmezci/ozu/segment-analysis"
        #base_folder_name = "/Users/mehmetpekmezci/ozu/segment-analysis/fine-tune"
        dir_arr = [name for name in os.listdir(base_folder_name) if
                   name.startswith("segment-size")]
        for dir_name in dir_arr:
            dir_path = f"{base_folder_name}/{dir_name}"
            path = f"{dir_path}/cnmp_best_validation.h5"
            first_segment_size = int(dir_name.split("-")[2])
            node_count = int(dir_name.split("-")[3])
            segment_borders = [[0, first_segment_size], [first_segment_size, 50]]
            trainer = CCnmpTrainer(data_x, data_y, data_x, data_y, segment_count=len(segment_borders),
                                   segment_borders=segment_borders)
            x_arr = trainer.get_x_arr_new()
            y_arr = trainer.get_y_arr_new()

            model1 = CCnmp2(trainer.d_x, trainer.d_y, node_count=node_count,
                            segment_count=len(segment_borders)).double()
            model1.load_state_dict(torch.load(path))

            obs_idx_arr = []
            for i in range(first_segment_size):
                for j in range(50 - first_segment_size):
                    obs_idx_arr.append((i, j))

            error_list = []
            for obs_idx in obs_idx_arr:
                cumulative_offset = 0
                observation_error = 0
                for i in range(data_x.shape[0]):
                    obs_arr = []
                    for j in range(trainer.segment_count):
                        obs_arr.append(np.array([np.concatenate((x_arr[j][i][obs_idx[j]], y_arr[j][i][obs_idx[j]]))]))
                    p, p_std = trainer.predict_model(model1, obs_arr, x_arr, plot=False, training_traj_id=i,
                                                     model_name=dir_name, targets=[x_arr[0][0][:], x_arr[1][0][:]])
                    traj_error = np.mean(np.sqrt((p - data_y[i]) ** 2))
                    observation_error += traj_error
                observation_error = observation_error / data_x.shape[0]
                error_list.append(observation_error)

            print(f"{dir_name} mean: {np.mean(error_list)}, std: {np.std(error_list)}")

            if first_segment_size in error_dict:
                error_dict.get(first_segment_size).append(np.mean(error_list))
            else:
                error_dict[first_segment_size] = [np.mean(error_list)]

        plt.subplots(1, 1, figsize=(16, 19))
        bar_size = 0.6
        key_list = list(error_dict.keys())
        min_key = min(error_dict, key=lambda k: np.mean(error_dict[k]))
        if len(key_list) > 1:
            bar_size = (max(key_list) - min(key_list)) * 0.6 / len(key_list)
        xticks= []
        for key in key_list:
            mean_err = np.mean(error_dict[key])
            print(f"first segment size : {key}, mean errors : {error_dict[key]}")
            color="C1" if key == min_key else "C0"
            plt.bar(key, mean_err, color=color, width=bar_size)
            xticks.append(int(key))
        plt.xlabel("First Segment Length", fontsize=25)
        plt.ylabel("Mean Error", fontsize=25)
        plt.title("Prediction Errors For Different Segment Sizes", fontsize=30)
        plt.tick_params(labelsize=20)
        #plt.xticks(np.arange(10)*5,np.arange(10)*5)
        plt.xticks(xticks, xticks)
        plt.savefig(f'{base_folder_name}/errors_by_segment_length_fine_tune.png')
