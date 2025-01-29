import pickle
import numpy as np

from trainer import CCnmpTrainer, CCnmp2

if __name__ == "__main__":

    node_count = 16
    first_segment_length=20

    file = open("../input/synthetic_dataset.dat", 'rb')
    data_set = pickle.load(file)
    file.close()
    trajectories = np.zeros((4, 50))
    for i, trajectory in enumerate(data_set[0]):
        trajectories[i] = trajectory[:, 1]

    y_arr = []
    x_arr = []

    for trajectory in trajectories:
        x_arr.append(np.linspace(0, 1, len(trajectory)).reshape((-1, 1)))
        y_arr.append(trajectory.reshape((-1, 1)))

    data_x = np.array(x_arr)
    data_y = np.array(y_arr)


    segment_borders = [[0,first_segment_length], [first_segment_length, 50]]

    trainer = CCnmpTrainer(data_x, data_y, data_x, data_y, segment_borders=segment_borders, segment_count=len(segment_borders))
    model = CCnmp2(trainer.d_x, trainer.d_y, node_count=node_count, segment_count=len(segment_borders)).double()

    total_params = 0
    for p in model.parameters():
        param_count = p.numel()
        total_params+= param_count
        #print(p)

    print(f'Total number of parameters: {total_params}')