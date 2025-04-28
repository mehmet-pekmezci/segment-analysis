import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import ticker

from trainer import CCnmpTrainer, CCnmp2


def map_indexes(observation_indexes_in, segment_borders_in):
    results = []
    for obs_index_arr in observation_indexes_in:
        result = [[] for _ in segment_borders_in]
        for obs_index in obs_index_arr:
            for segment_id, (start, end) in enumerate(segment_borders_in):
                if start <= obs_index < end:
                    result[segment_id].append(obs_index - start)
                    break
        results.append(result)
    return results


def get_random_indexes(border_map_in):
    random_indexes = []
    for start, end in border_map_in["four"]:
        random_indexes.append(np.random.randint(start, end))
    for start, end in border_map_in["five"]:
        found = False
        for i in range(start, end):
            if i in random_indexes:
                found = True
        if not found:
            random_indexes.append(np.random.randint(start, end))
    return random_indexes


def find_outliers(errors_in):
    sorted_arr = np.sort(errors_in)
    inc_arr = [sorted_arr[i] - sorted_arr[i-1] for i in range(1, len(sorted_arr))]
    index = np.argmax(inc_arr)
    if inc_arr[index] > 3* (max(sorted_arr) - min(sorted_arr)) / len(sorted_arr):
        return sorted_arr [index+1:].tolist()
    return []


if __name__ == "__main__":
    file = open("../input/synthetic_dataset.dat", 'rb')
    data_set = pickle.load(file)
    file.close()
    segment_border_map = {}
    segment_border_map["two"] = [[0,25], [25, 50]]
    segment_border_map["three"] = [[0,17], [17, 34], [34, 50]]
    segment_border_map["four"] = [[0,12], [12, 24], [24, 36], [36, 50]]
    segment_border_map["five"] = [[0, 10], [10, 20], [20, 30], [30, 40], [40, 50]]

    random_obs_test_count = 1000

    trajectories = np.zeros((4, 50))
    for i, trajectory in enumerate(data_set[0]):
        trajectories[i] = trajectory[:, 1]


    trial_id = time.time()
    y_arr = []
    x_arr = []

    for trajectory in trajectories:
        x_arr.append(np.linspace(0, 1, len(trajectory)).reshape((-1, 1)))
        y_arr.append(trajectory.reshape((-1, 1)))

    data_x = np.array(x_arr)
    data_y = np.array(y_arr)

    observation_indexes = []
    while len(observation_indexes) < random_obs_test_count:
        indexes = get_random_indexes(segment_border_map)
        if len(indexes) == 5 and indexes not in observation_indexes:
            observation_indexes.append(indexes)

    error_dict = {}
    base_folder_name = "/Users/mehmetpekmezci/ozu/segment-analysis"
    dir_arr = []
    for name in os.listdir(base_folder_name):
        for prefix in segment_border_map.keys():
            if name.startswith(f"{prefix}-segment-"):
                dir_arr.append(name)

    error_map = {}
    for dir_name in dir_arr:
        print(f'Evaluating {dir_name}')
        dir_path = f"{base_folder_name}/{dir_name}"
        path = f"{dir_path}/cnmp_best_validation.h5"
        key = dir_name.split("-")[0]
        node_count = int(dir_name.split("-")[-3])
        trial = int(dir_name.split("-")[-1])
        segment_borders = segment_border_map[key]
        trainer = CCnmpTrainer(data_x, data_y, data_x, data_y, segment_count=len(segment_borders),
                               segment_borders=segment_borders)
        x_arr = trainer.get_x_arr_new()
        y_arr = trainer.get_y_arr_new()

        model1 = CCnmp2(trainer.d_x, trainer.d_y, node_count=node_count,
                        segment_count=len(segment_borders)).double()
        model1.load_state_dict(torch.load(path))

        targets = []
        for x_element in x_arr:
            targets.append(x_element[0][:])

        mapped_indexes = map_indexes(observation_indexes, segment_borders)
        model_errors = []
        for traj_id in range(data_x.shape[0]):
            traj_errors = []
            for indexes in mapped_indexes:
                obs_arr = []
                for segment_index, time_indexes in enumerate(indexes):
                    observations = []
                    for time_index in time_indexes:
                        observations.append(np.concatenate((x_arr[segment_index][traj_id][time_index], y_arr[segment_index][traj_id][time_index])))
                    obs_arr.append(np.array(observations))

                p, p_std = trainer.predict_model(model1, obs_arr, x_arr, plot=False, training_traj_id=traj_id,
                                                 model_name=dir_name, targets=targets)
                pred_error = np.mean(np.sqrt((p - data_y[traj_id]) ** 2))
                traj_errors.append(pred_error)
            print(f"traj[{traj_id}]_error : {np.average(traj_errors)}")
            model_errors.append(np.average(traj_errors))
        print(f"model[{key}][{trial}]_error: {np.average(model_errors)}")
        if key not in error_map:
            error_map[key] = [np.average(model_errors)]
        else:
            error_map[key].append(np.average(model_errors))

    fig, axs = plt.subplots(1, 3, figsize=(19,10))
    colors = ["orange", "blue", "red", "green", "magenta"]
    max_err = 0
    all_outlier_indexes = []
    all_outlier_values = []
    for key, value in segment_border_map.items():
        seg_count = len(value)
        errors = error_map[key]
        outliers = find_outliers(errors)
        without_outliers = a = [x for x in errors if x not in outliers]
        all_outlier_values += outliers
        all_outlier_indexes += [seg_count] * len(outliers)
        axs[0].scatter([seg_count] * len(errors), errors, color=colors[seg_count-1], label=f"Average error of trial with {seg_count} segments")
        axs[1].bar(seg_count, np.average(errors), color=colors[seg_count-1])
        axs[1].errorbar(seg_count, np.average(errors), yerr=np.std(errors), fmt='o-', capsize=5, label="Mean ± Std Dev", color="black")
        axs[1].text(seg_count, np.average(errors) + np.std(errors) + 0.005, f"{np.average(errors):.3f}",
                ha='center', fontsize=10, fontweight='bold')

        print(f"without outliers comparison : {np.sort(errors)[:-2]} - {np.sort(without_outliers)}")
        axs[2].bar(seg_count, np.average(without_outliers), color=colors[seg_count-1])
        axs[2].errorbar(seg_count, np.average(without_outliers), yerr=np.std(without_outliers), fmt='o-', capsize=5, label="Mean ± Std Dev", color="black")
        axs[2].text(seg_count, np.average(without_outliers) + np.std(without_outliers) + 0.005, f"{np.average(without_outliers):.3f}",
                    ha='center', fontsize=10, fontweight='bold')

        max_err = max(max_err, max(errors))

    axs[0].scatter(all_outlier_indexes, all_outlier_values, marker='o', s=200, lw=5, facecolors='none', edgecolors='black', label="Outlier")

    print(f'outliers : {all_outlier_indexes} - {all_outlier_values}')
    for ax in axs:
        ax.set_xticks(np.arange(2,6, dtype=int))
        #ax.set_xticklabels(["two", "three", "four", "five"])
        ax.set_xticklabels(["2", "3", "4", "5"])
        ax.set_xlabel('Segment Count', fontsize=15)
        ax.tick_params(labelsize=12)
        ax.set_ylim(0, max_err+0.01)
    axs[0].set_ylabel('Error', fontsize=15)
    axs[0].set_title('Avg. Error Per Trial', fontsize=20)
    axs[1].set_title('Overall Avg. Error', fontsize=20)
    axs[2].set_title('Overall Avg. Error Discarding Outliers', fontsize=20)

    axs[0].legend(loc="upper left")

    plt.suptitle("Avg. Prediction Errors For Different Segment Counts", fontsize=30)

    plt.savefig("/Users/mehmetpekmezci/ozu/segment-analysis/different_segment_count.png")


