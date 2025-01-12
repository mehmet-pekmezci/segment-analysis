import os
import time
import pickle

import numpy as np
import torch

from trainer.trainer import CCnmpTrainer, CCnmp2

if __name__ == "__main__":
    training_iter_count = int(os.environ['training_iters'])
    validation_iter_count = int (training_iter_count / 100)
    file = open("../input/synthetic_dataset.dat", 'rb')
    data_set = pickle.load(file)
    file.close()
    trajectories = np.zeros((4, 50))
    for i, trajectory in enumerate(data_set[0]):
        trajectories[i] = trajectory[:, 1]

    torch.set_num_threads(50)
    trial_id = time.time()
    node_count = 16
    first_segment_lengths = [5, 10, 15, 20, 25, 30, 35, 40, 45]

    for iter, first_segment_length in enumerate(first_segment_lengths):
        y_arr = []
        x_arr = []

        for trajectory in trajectories:
            x_arr.append(np.linspace(0, 1, len(trajectory)).reshape((-1, 1)))
            y_arr.append(trajectory.reshape((-1, 1)))

        data_x = np.array(x_arr)
        data_y = np.array(y_arr)

        dir_name= f"../output/segment-size-{first_segment_length}-{node_count}-{trial_id}"
        os.mkdir(dir_name)
        segment_borders = [[0,first_segment_length], [first_segment_length, 50]]

        trainer = CCnmpTrainer(data_x, data_y, data_x, data_y, segment_borders=segment_borders, segment_count=len(segment_borders))
        model1 = CCnmp2(trainer.d_x, trainer.d_y, node_count=node_count, segment_count=len(segment_borders)).double()

        print(f'Training starting for {dir_name}')
        start = time.time()
        trainer.train(model1, dir_name, iter_count=training_iter_count, validation_checkpoint=validation_iter_count, lr_decay_rate=0.999)
        end = time.time()
        print(f'Training Complete in {end-start}')
