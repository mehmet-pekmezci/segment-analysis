import torch


def print_angles(file_handle, joint_angles):
    if type(joint_angles) == torch.Tensor:
        joint_angles = joint_angles.numpy()

    for angle in joint_angles:
        file_handle.write("," + str(round(angle, 3)))


def print_ccnmp_trajectory(file_path, left, right, left_end=None, right_start=2, ms_interval = 50):
    counter=0
    time_in_ms=0
    arm_no=1
    if left_end is None:
        left_end = len(left)

    with open(file_path, 'w') as file:
        for i in range(0, left_end):
            file.write(f"{counter}/{time_in_ms}/setdesq/ {arm_no}/ 0.000,0.000")
            print_angles(file, left[i])
            file.write(",0.000\n")
            counter+=1
            time_in_ms += ms_interval

        for i in range(right_start, len(right)):
            file.write(f"{counter}/{time_in_ms}/setdesq/ {arm_no}/ 0.000,0.000")
            print_angles(file, right[i])
            file.write(",0.000\n")
            counter += 1
            time_in_ms += ms_interval

def print_cnmp_trajectory(file_path, trajectory, ms_interval = 50):
    counter=0
    time_in_ms=0
    arm_no=1

    with open(file_path, 'w') as file:
        for step in trajectory:
            file.write(f"{counter}/{time_in_ms}/setdesq/ {arm_no}/ 0.000,0.000")
            print_angles(file, step)
            file.write(",0.000\n")
            counter+=1
            time_in_ms += ms_interval

