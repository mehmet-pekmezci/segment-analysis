import os
import time
import pickle

import numpy as np
import torch

from trainer import CCnmpTrainer, CCnmp2

if __name__ == "__main__":
    training_iter_count = int(os.getenv('training_iters', 1_000_000))
    first_segment_length = int(os.getenv('first_segment_length', 10))
    second_segment_length = int(os.getenv('second_segment_length', 10))
    third_segment_length = int(os.getenv('third_segment_length', 10))
    fourth_segment_length = int(os.getenv('fourth_segment_length', 10))
    fifth_segment_length = 50 - first_segment_length - second_segment_length - third_segment_length-fourth_segment_length
    repeat_count = int(os.getenv('repeat_count', 1))
    node_count = int(os.getenv('node_count', 7))
    validation_iter_count = int (training_iter_count / 100)
    file = open("../input/synthetic_dataset.dat", 'rb')
    data_set = pickle.load(file)
    file.close()
    trajectories = np.zeros((4, 50))
    for i, trajectory in enumerate(data_set[0]):
        trajectories[i] = trajectory[:, 1]

    torch.set_num_threads(50)
    trial_id = time.time()

    for iter in range(repeat_count):
        y_arr = []
        x_arr = []

        for trajectory in trajectories:
            x_arr.append(np.linspace(0, 1, len(trajectory)).reshape((-1, 1)))
            y_arr.append(trajectory.reshape((-1, 1)))

        data_x = np.array(x_arr)
        data_y = np.array(y_arr)

        dir_name= f"../output/five-segment-{first_segment_length}-{second_segment_length}-{third_segment_length}-{fourth_segment_length}-{fifth_segment_length}-{node_count}-{trial_id}-{iter}"
        os.mkdir(dir_name)
        third_segment_start = first_segment_length + second_segment_length
        fourth_segment_start = third_segment_start + third_segment_length
        fifth_segment_start = fourth_segment_start + fifth_segment_length
        segment_borders = [[0,first_segment_length],
                           [first_segment_length, third_segment_start],
                           [third_segment_start, fourth_segment_start],
                           [fourth_segment_start, fifth_segment_start],
                           [fifth_segment_start, 50]]

        trainer = CCnmpTrainer(data_x, data_y, data_x, data_y, segment_borders=segment_borders, segment_count=len(segment_borders))
        model1 = CCnmp2(trainer.d_x, trainer.d_y, node_count=node_count, segment_count=len(segment_borders)).double()



        total_params = 0
        for p in model1.parameters():
            param_count = p.numel()
            total_params+= param_count
            print(p)
        print(f'Total number of parameters: {total_params}')
        print(f'Training starting for {dir_name}')
        start = time.time()
        trainer.train(model1, dir_name, iter_count=training_iter_count, validation_checkpoint=validation_iter_count, lr_decay_rate=0.999)
        end = time.time()
        print(f'Training Complete in {end-start}')
