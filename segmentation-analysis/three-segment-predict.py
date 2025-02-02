import os
import pickle
import time

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

        #dir_arr = [name for name in os.listdir("../output") if name.startswith("three-segment") ]
        dir_arr = [name for name in os.listdir("/Users/mehmetpekmezci/ozu/segment-analysis") if name.startswith("three-segment") ]
        for dir_name in dir_arr:
            dir_path= f"/Users/mehmetpekmezci/ozu/segment-analysis/{dir_name}"
            #dir_path= f"../output/{dir_name}"
            path= f"{dir_path}/cnmp_best_validation.h5"
            first_segment_size = int(dir_name.split("-")[2])
            second_segment_size = int(dir_name.split("-")[3])
            node_count = int(dir_name.split("-")[4])
            node_count=17
            third_segment_size = 50 - first_segment_size - second_segment_size

            second_segment_start = first_segment_size
            third_segment_start = second_segment_start + second_segment_size

            segment_borders = [[0,first_segment_size],
                               [second_segment_start, third_segment_start],
                               [third_segment_start, 50]]
            trainer = CCnmpTrainer(data_x, data_y, data_x, data_y, segment_count=len(segment_borders), segment_borders=segment_borders)
            x_arr = trainer.get_x_arr_new()
            y_arr = trainer.get_y_arr_new()

            model1 = CCnmp2(trainer.d_x, trainer.d_y, node_count=node_count, segment_count=len(segment_borders)).double()
            model1.load_state_dict(torch.load(path))


            total_params = 0
            for p in model1.parameters():
                param_count = p.numel()
                total_params+= param_count
                print(p)
            print(f'Total number of parameters: {total_params}')
            obs_idx_arr = [int(first_segment_size/2),
                           int(second_segment_size/2) + second_segment_start,
                           int(third_segment_size/2) + third_segment_start]
            for idx in range(1):
                cumulative_offset = 0
                for j in range(trainer.segment_count):
                    if j>0:
                        cumulative_offset += len(x_arr[j-1][0])
                        obs_idx_arr[j] -= cumulative_offset

                for i in range(data_x.shape[0]):
                    obs_arr = []
                    for j in range(trainer.segment_count):
                        obs_idx = obs_idx_arr[j]
                        obs_arr.append(np.array([np.concatenate((x_arr[j][i][obs_idx], y_arr[j][i][obs_idx]))]))
                    targets = []
                    for x_element in x_arr:
                        targets.append(x_element[0][:])
                    p, p_std = trainer.predict_model(model1, obs_arr, x_arr, plot=True, training_traj_id=i, model_name=dir_name, plot_path=f"{dir_path}/trial-{idx}-traj-{i}.png", targets= targets)
                    print_cnmp_trajectory(f"{dir_path}/trial-{idx}-traj-{i}.txt",p)
