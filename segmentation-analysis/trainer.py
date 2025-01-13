import os

import numpy as np
import math
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
import time

class CCnmp(nn.Module):
    def __init__(self, d_x, d_y, node_count = 128, segment_count = 1):
        super(CCnmp, self).__init__()
        self.encoder_arr = nn.ModuleList()
        self.decoder_arr = nn.ModuleList()
        self.segment_count = segment_count
        # Encoder takes observations which are (X,Y) tuples and produces latent representations for each of them
        for i in range (segment_count):
            encoder = nn.Sequential(
                nn.Linear(d_x + d_y, node_count), nn.ReLU(),
                nn.Linear(node_count, node_count), nn.ReLU(),
                nn.Linear(node_count, node_count)
            )
            self.encoder_arr.append(encoder)

        # Decoder takes the (r_mean, target_t) tuple and produces mean and std values for each dimension of the output
        for i in range(segment_count):
            decoder = nn.Sequential(
                nn.Linear(segment_count * node_count + d_x, node_count), nn.ReLU(),
                nn.Linear(node_count, node_count), nn.ReLU(),
                nn.Linear(node_count, 2 * d_y)
            )
            self.decoder_arr.append(decoder)

    def forward(self, observations, target_t):
        r_mean_arr = []
        for i in range(self.segment_count):
            r_mean_arr.append(torch.mean(self.encoder_arr[i](observations[i]), dim=0))
        # combine all observations
        r_mean_combined = torch.cat(r_mean_arr, dim=0)

        outputs = []
        for i in range(self.segment_count):
            outputs.append(self.decoder_arr[i](torch.cat((r_mean_combined.repeat(target_t[i].shape[0], 1), target_t[i]), dim=-1)))
        return outputs

class CCnmp2(nn.Module):
    def __init__(self, d_x, d_y, node_count = 128, segment_count = 1):
        super(CCnmp2, self).__init__()
        self.encoder_arr = nn.ModuleList()
        self.decoder_arr = nn.ModuleList()
        self.segment_count = segment_count
        # Encoder takes observations which are (X,Y) tuples and produces latent representations for each of them
        for i in range (segment_count):
            encoder = nn.Sequential(
                nn.Linear(d_x + d_y, node_count), nn.ReLU(),
                nn.Linear(node_count, int(node_count/2)), nn.ReLU(),
                nn.Linear(int(node_count/2), int(node_count/4))
            )
            self.encoder_arr.append(encoder)

        # Decoder takes the (r_mean, target_t) tuple and produces mean and std values for each dimension of the output
        for i in range(segment_count):
            decoder = nn.Sequential(
                nn.Linear(segment_count * int(node_count/4) + d_x, int(node_count/2)), nn.ReLU(),
                nn.Linear(int(node_count/2), node_count), nn.ReLU(),
                nn.Linear(node_count, 2 * d_y)
            )
            self.decoder_arr.append(decoder)

    def forward(self, observations, target_t):
        r_mean_arr = []
        for i in range(self.segment_count):
            r_mean_arr.append(torch.mean(self.encoder_arr[i](observations[i]), dim=0))
        # combine all observations
        r_mean_combined = torch.cat(r_mean_arr, dim=0)

        outputs = []
        for i in range(self.segment_count):
            outputs.append(self.decoder_arr[i](torch.cat((r_mean_combined.repeat(target_t[i].shape[0], 1), target_t[i]), dim=-1)))
        return outputs

def log_prob_loss(outputs, target):

    chunked_outputs = [output.chunk(2, dim=-1) for output in outputs]
    mean_arr = []
    sigma_arr = []
    [mean_arr.append(element[0]) for element in chunked_outputs]
    [sigma_arr.append(element[1]) for element in chunked_outputs]
    mean = torch.cat(mean_arr)
    sigma = torch.cat(sigma_arr)
    sigma = F.softplus(sigma)
    dist = D.Independent(D.Normal(loc=mean, scale=sigma), 1)
    return -torch.mean(dist.log_prob(target))

class CCnmpTrainer:
    def __init__(self, X, Y, v_X, v_Y, segment_count, segment_borders=None):
        self.X = X
        self.Y = Y
        self.v_X = v_X
        self.v_Y = v_Y

        self.obs_max = 5
        self.d_N = X.shape[0]
        self.d_x, self.d_y = (X.shape[-1], Y.shape[-1])
        self.time_len = X.shape[1]
        self.segment_count = segment_count
        self.x_offset_arr = np.arange(segment_count, dtype=float)
        self.vx_offset_arr = np.arange(segment_count, dtype=float)
        self.x_offset_arr[0] = 0
        self.vx_offset_arr[0] = 0
        self.segment_borders=segment_borders

    def train(self, model, path, iter_count=5_000_000, validation_checkpoint=100_000, loss_checkpoint=1000, lr_decay_rate=0.99):
        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=lr_decay_rate)

        smooth_losses = [0]
        losses = []
        #loss_checkpoint = 1000
        plot_checkpoint = 5000055
        #validation_checkpoint = 100
        validation_error = 9999999
        plot_obs_and_target=False
        last_progress = 0
        should_update_lr=False

        x_arr = self.get_x_arr_new()
        y_arr = self.get_y_arr_new()
        vx_arr = self.get_vx_arr_new()
        vy_arr = self.get_vy_arr_new()

        observation_duration_sum = 0
        fw_run_duration_sum = 0
        loss_calculation_duration_sum = 0
        backprop_duration_sum = 0
        for step in range(iter_count):  # loop over the dataset multiple times
            start = time.time_ns()
            observations = []
            targets = []
            target_outputs =[]

            obs_arr_x = []
            obs_arr_y = []
            tgt_arr_x = []
            tgt_arr_y = []
            d = np.random.randint(0, self.d_N)
            for i in range(self.segment_count):
                observations_i, target_t_i, target_output_i = self.get_train_sample(x_arr[i], y_arr[i], d, 5)
                observations.append(observations_i)
                targets.append(target_t_i)
                target_outputs.append(target_output_i)

                if plot_obs_and_target:

                    obs_i_x_arr = []
                    obs_i_y_arr = []
                    for obs in observations_i:
                        obs_i_x_arr.append(obs[0].item())
                        obs_i_y_arr.append(obs[1].item())
                    obs_arr_x.append(obs_i_x_arr)
                    obs_arr_y.append(obs_i_y_arr)

                    tgt_i_x_arr = []
                    for tgt in target_t_i:
                        tgt_i_x_arr.append(tgt[0].item())
                    tgt_arr_x.append(tgt_i_x_arr)

                    tgt_i_y_arr = []
                    for out in target_output_i:
                        tgt_i_y_arr.append(out[0].item())
                    tgt_arr_y.append(tgt_i_y_arr)

            observations_found = time.time_ns()
            observation_duration_sum += observations_found - start

            optimizer.zero_grad()
            outputs = model(observations, targets)
            fw_run_complete = time.time_ns()
            fw_run_duration_sum += fw_run_complete - observations_found
            target = torch.cat(target_outputs, dim=0)
            loss = log_prob_loss(outputs, target)
            loss_calculated = time.time_ns()
            loss_calculation_duration_sum += fw_run_complete - loss_calculated

            loss.backward()
            optimizer.step()

            backprop_complete = time.time_ns()
            backprop_duration_sum += backprop_complete - loss_calculated

            if step % loss_checkpoint == 0:
                losses.append(loss.data)
                smooth_losses[-1] += loss.data / (plot_checkpoint / loss_checkpoint)

            if step > 0 and step % validation_checkpoint == 0:
                print(f"step : {step}, obs: {round(observation_duration_sum/step, 4)}, fw: {round(fw_run_duration_sum/step, 4)}, loss: {round(loss_calculation_duration_sum/step, 4)}, bp: {round(backprop_duration_sum/step, 4)}")
                if plot_obs_and_target:
                    for i in range(len(obs_arr_x)):
                        plt.scatter(obs_arr_x[i], obs_arr_y[i], label=f'obs{i}')
                    for i in range(len(tgt_arr_x)):
                        plt.scatter(tgt_arr_x[i], tgt_arr_y[i], label=f'tgt{i}', marker='X')
                    plt.legend()
                    plt.show()

                current_error = 0
                for i in range(self.v_X.shape[0]):
                    obs_arr = []
                    vx_target=[]
                    for j in range(self.segment_count):
                        obs_arr.append(np.array([np.concatenate((vx_arr[j][i][0], vy_arr[j][i][0]))]))
                        for vx_segment in vx_arr:
                            for vx_point in vx_segment[i]:
                                vx_target.append(vx_point)

                    #predicted_Y, predicted_std = self.predict_model(model, obs_arr, np.concatenate(np.array(vx_arr)[:, i]), plot=False)
                    predicted_Y, predicted_std = self.predict_model(model, obs_arr, np.array(vx_target), plot=False)
                    # todo: predicted_Y, predicted_std = self.predict_model(model, obs_arr, self.v_X[i], plot=False)
                    current_error += np.mean(np.sqrt((predicted_Y - self.v_Y[i, :]) ** 2))
                if current_error < validation_error:
                    validation_error = current_error
                    torch.save(model.state_dict(), f'{path}/cnmp_best_validation.h5')
                    print('======================> New validation best. Error is ', current_error)
                    last_progress = step
                else:
                    if (step-last_progress) > 3 * validation_checkpoint:
                        if should_update_lr:
                            lr_scheduler.step()
                            print(f'step: {step + 1}, learning rate {lr_scheduler.get_last_lr()[0]}')
                            should_update_lr = False
                        else:
                            should_update_lr = True

            if step != 0 and step % plot_checkpoint == 0:
                print(step)
                # plotting training examples and smoothed losses

                plt.figure(figsize=(15, 5))
                plt.subplot(121)
                plt.title('Train Loss')
                plt.plot(range(len(losses)), losses)
                plt.subplot(122)
                plt.title('Train Loss (Smoothed)')
                plt.plot(range(len(smooth_losses)), smooth_losses)
                plt.show()

                # plotting validation cases
                for i in range(self.v_X.shape[0]):

                    obs_arr = []
                    for j in range(self.segment_count):
                        obs_arr.append(np.array([np.concatenate((vx_arr[j][i][0], vy_arr[j][i][0]))]))
                    self.predict_model(model, obs_arr, np.concatenate(np.array(vx_arr)[:, i]), plot=True)
                    # todo
                if step != 0:
                    smooth_losses.append(0)
        print('Finished Training')

    def get_vy_arr(self):
        vy_arr = np.array_split(self.v_Y, self.segment_count, axis=1)
        return vy_arr

    def get_vy_arr_new(self):
        if self.segment_borders is None:
            vy_arr = np.array_split(self.v_Y, self.segment_count, axis=1)
        else:
            vy_arr = []
            for segment_border in self.segment_borders:
                vy_arr.append(self.v_Y[:, segment_border[0]:segment_border[1], :])

        return vy_arr

    def get_vx_arr(self):
        vx_arr = np.array_split(self.v_X, self.segment_count, axis=1)
        for i in range(1, self.segment_count):
            self.vx_offset_arr[i] = vx_arr[i].min()
            vx_arr[i] = vx_arr[i] - self.vx_offset_arr[i]
        return vx_arr
    def get_vx_arr_new(self):
        if self.segment_borders is None:
            vx_arr = np.array_split(self.v_X, self.segment_count, axis=1)
        else:
            vx_arr = []
            for segment_border in self.segment_borders:
                vx_arr.append(self.v_X[:, segment_border[0]:segment_border[1], :])
        for i in range(1, self.segment_count):
            self.vx_offset_arr[i] = vx_arr[i].min()
            vx_arr[i] = vx_arr[i] - self.vx_offset_arr[i]
        return vx_arr

    def get_y_arr(self):
        y_arr = np.array_split(self.Y, self.segment_count, axis=1)
        return y_arr

    def get_y_arr_new(self):
        if self.segment_borders is None:
            y_arr = np.array_split(self.Y, self.segment_count, axis=1)
        else:
            y_arr = []
            for segment_border in self.segment_borders:
                y_arr.append(self.Y[:, segment_border[0]:segment_border[1], :])
        return y_arr

    def get_x_arr(self):
        x_arr = np.array_split(self.X, self.segment_count, axis=1)
        for i in range(1, self.segment_count):
            self.x_offset_arr[i] = x_arr[i].min()
            x_arr[i] = x_arr[i] - self.x_offset_arr[i]
        return x_arr
    def get_x_arr_new(self):
        if self.segment_borders is None:
            x_arr = np.array_split(self.X, self.segment_count, axis=1)
        else:
            x_arr = []
            for segment_border in self.segment_borders:
                x_arr.append(self.X[:, segment_border[0]:segment_border[1], :])
        for i in range(1, self.segment_count):
            self.x_offset_arr[i] = x_arr[i].min()
            x_arr[i] = x_arr[i] - self.x_offset_arr[i]
        return x_arr

    def get_concatenated_y_array(self, segment_predictions):
        if self.segment_borders is None:
            return np.concatenate(segment_predictions)
        else:
            y_arr = []
            y_arr.append(segment_predictions[0])
            for i in range(1, self.segment_count):
                y_arr.append(segment_predictions[i][self.segment_borders[i-1][1]-self.segment_borders[i][0]:])
            return np.concatenate(y_arr)

    def get_train_sample(self, X_in, Y_in, d, obs_max_in=None):
        obs_max_in = self.obs_max if obs_max_in is None else obs_max_in
        time_length_in = len(X_in[0])

        n = np.random.randint(0, obs_max_in) + 1

        while n >= time_length_in:
            n-=1
        observations = np.zeros((n, self.d_x + self.d_y))
        target_X = np.zeros((1, self.d_x))
        target_Y = np.zeros((1, self.d_y))

        perm = np.random.permutation(time_length_in)
        observations[:n, :self.d_x] = X_in[d, perm[:n]]
        observations[:n, self.d_x:self.d_x + self.d_y] = Y_in[d, perm[:n]]
        target_X[0] = X_in[d, perm[n]]
        target_Y[0] = Y_in[d, perm[n]]
        return torch.from_numpy(observations), torch.from_numpy(target_X), torch.from_numpy(target_Y)

    def predict_model(self, model, observations, target_X, plot=True, training_traj_id=None, model_name=None, plot_path=None, targets=None):
        with torch.no_grad():
            obs = [torch.from_numpy(observation) for observation in observations]
            if self.segment_borders is None:
                targets = np.array_split(target_X, self.segment_count )
            else:
                if targets is None:
                    targets = []
                    for sb in self.segment_borders:
                        targets.append(target_X[sb[0]:sb[1], :])
            tgt = [torch.from_numpy(target) for target in targets]
            outputs = model(obs, tgt)
        predictions =[output.numpy() for output in outputs]
        predicted_Y_arr = [p[:, :self.d_y] for p in predictions]
        predicted_std_arr = [p[:, self.d_y:] for p in predictions]
        predicted_Y = self.get_concatenated_y_array(predicted_Y_arr)
        predicted_std_raw = self.get_concatenated_y_array(predicted_std_arr)
        predicted_std = np.log(1 + np.exp(predicted_std_raw))

        # plt.plot(np.arange(len(predicted_Y_arr[0])), predicted_Y_arr[0])
        # plt.plot(np.arange(len(predicted_Y_arr[0]), 50), predicted_Y_arr[1])
        # plt.plot(np.arange(50),  self.Y[training_traj_id, :, 0], alpha=0.25, linewidth=10, zorder=5)
        # plt.plot()
        # plt.show()

        title = "Predicted Training Trajectory"
        if training_traj_id is not None:
            error = np.mean(np.sqrt((predicted_Y - self.Y[training_traj_id, :]) ** 2))
            title = f"Predicted Training Trajectory #{training_traj_id}, avg error: {round(error, 4)}"
        if model_name is not None:
            title += f" ({model_name})"
        if plot:  # We highly recommend that you customize your own plot function, but you can use this function as default
            n_cols = min(3, self.d_y)
            n_rows = math.ceil(self.d_y / n_cols)

            fig, axs = plt.subplots(n_rows, n_cols, figsize=(12,19))
            for i in range(self.d_y):  # for every feature in Y vector we are plotting training data and its prediction
                if n_rows == 1:
                    if n_cols == 1:
                        ax = axs
                    else:
                        ax = axs[i % n_cols]
                else:
                    ax = axs[math.floor(i/n_cols), i%n_cols]
                j = 0
                for j in range(self.d_N):
                    if training_traj_id is None:
                        ax.plot(self.X[j, :, 0], self.Y[j, :, i])
                    elif training_traj_id == j:
                        ax.plot(self.X[j, :, 0], self.Y[j, :, i], color='green', alpha=0.25, linewidth=10, zorder=5)
                ax.plot(self.X[j, :, 0], predicted_Y[:, i], color='black')
                ax.errorbar(self.X[j, :, 0], predicted_Y[:, i], yerr=predicted_std[:, i], color='black', alpha=0.4)
                for j in range(len(observations)):
                    ax.scatter(observations[j][:, 0]+self.x_offset_arr[j], observations[j][:, self.d_x + i], marker="X", color='black')
                ax.set_title(f"dim-{i}")
            plt.suptitle(title)
            if plot_path is None:
                plt.show()
            else:
                plt.savefig(plot_path)
                plt.close(fig)
        return predicted_Y, predicted_std

def dist_generator(d, x, param, noise=0):
    f = (math.exp(-x ** 2 / (2. * param[0] ** 2)) / (math.sqrt(2 * math.pi) * param[0])) + param[1]
    return f + (noise * (np.random.rand() - 0.5) / 100.)


def generate_demonstrations(time_len=200, params=None, title=None):
    fig = plt.figure(figsize=(5, 5))
    x = np.linspace(-0.5, 0.5, time_len)
    times = np.zeros((params.shape[0], time_len, 1))
    times[:] = x.reshape((1, time_len, 1)) + 0.5
    values = np.zeros((params.shape[0], time_len, 1))
    for d in range(params.shape[0]):
        for i in range(time_len):
            values[d, i] = dist_generator(d, x[i], params[d])
        plt.plot(times[d], values[d])
    plt.title(title + ' Demonstrations')
    plt.ylabel('Y')
    plt.xlabel('time (t)')
    plt.show()
    return times, values



if __name__ == "__main__":
    time_length = 16
    training_conditioning = [[0.6, -0.1], [0.5, -0.23], [0.4, -0.43], [-0.6, 0.1], [-0.5, 0.23], [-0.4, 0.43]]
    val_conditioning = [[0.55, -0.155], [0.45, -0.32], [-0.45, 0.32], [-0.55, 0.155]]

    #training_conditioning = [[0.6, -0.1]]
    #val_conditioning = [[0.6, -0.1]]
    X, Y = generate_demonstrations(time_len=time_length, params=np.array(
        training_conditioning), title='Training')
    v_X, v_Y = generate_demonstrations(time_len=time_length,
                                       params=np.array(val_conditioning),
                                       title='Validation')

    trainer = CCnmpTrainer(X, Y, v_X, v_Y, segment_count=4)
    model1 = CCnmp2(trainer.d_x, trainer.d_y, node_count=80, segment_count=4).double()
    os.rmdir("test_for_ccnmp")
    os.mkdir("test_for_ccnmp")
    trainer.train(model1, "./test_for_ccnmp", iter_count=100_000, validation_checkpoint=1000, loss_checkpoint=100)