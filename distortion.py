import os
import argparse
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import torchio as tio
from torchio.transforms import RandomMotion, RandomGhosting, RandomSpike, RandomBiasField, RandomNoise, RandomBlur


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def make_dirs():
    dics = [
        "./datasets/dist/",
        "./datasets/dist/OASIS_T1w/",
        "./datasets/dist/NFBS_T1w/",
        "./datasets/dist/IXI_T1w/",
        "./datasets/dist/IXI_T2w/",
        "./datasets/dist/QTAB_T1w/",
        "./datasets/dist/QTAB_T2w/",
        "./datasets/dist/ARC_T1w/",
        "./datasets/dist/ARC_T2w/",
        "./datasets/dist/ABIDE_ccs/",
        "./datasets/dist/ABIDE_cpac/",
        "./datasets/dist/ABIDE_dparsf/",
        "./datasets/dist/ABIDE_niak/"
    ]

    for dic in dics:
        os.makedirs(dic, exist_ok=True)



def random_motion_distortion(ds_type_combinations):
    # add random motion artifacts
    for ds_name, sub_type in ds_type_combinations:
        print(ds_name, sub_type)

        dataset_path_src = f"./datasets/src/{ds_name}_{sub_type}/"
        dataset_path_dist = f"./datasets/dist/{ds_name}_{sub_type}/"
        path_list = os.listdir(dataset_path_src)
        path_list.sort()

        trans = [1, 5, 10, 20]
        nums = [1, 3, 5, 10]
        levels = ['motion_1', 'motion_2', 'motion_3', 'motion_4']

        for level in levels:
            if not os.path.exists(os.path.join(dataset_path_dist, level)):
                os.mkdir(os.path.join(dataset_path_dist, level))

        count = 0
        for filename in path_list:
            data = np.load(os.path.join(dataset_path_src, filename))
            data = data.reshape(1, *data.shape)

            count += 1
            if count % 10 == 0:
                print('\t', ds_name, sub_type, count)

            for tran, num, level in zip(trans, nums, levels):
                data_dist = np.array(RandomMotion(translation=(tran-1, tran), num_transforms=num)(data))[0, :, :, :]
                np.save(os.path.join(dataset_path_dist, level, filename), data_dist)


def random_ghosting_distortion(ds_type_combinations):
    # add random ghosting artifacts
    for ds_name, sub_type in ds_type_combinations:
        print(ds_name, sub_type)

        dataset_path_src = f"./datasets/src/{ds_name}_{sub_type}/"
        dataset_path_dist = f"./datasets/dist/{ds_name}_{sub_type}/"
        path_list = os.listdir(dataset_path_src)
        path_list.sort()

        nums = [3, 5, 10, 20]
        ints = [0.8, 1.5, 2.5, 5]
        levels = ['ghosting_1', 'ghosting_2', 'ghosting_3', 'ghosting_4']

        for level in levels:
            if not os.path.exists(os.path.join(dataset_path_dist, level)):
                os.mkdir(os.path.join(dataset_path_dist, level))

        count = 0
        for filename in path_list:
            data = np.load(os.path.join(dataset_path_src, filename))
            data = data.reshape(1, *data.shape)

            count += 1
            if count % 10 == 0:
                print('\t', ds_name, sub_type, count)

            for num, intt, level in zip(nums, ints, levels):
                data_dist = np.array(RandomGhosting(num_ghosts=(num, num+2), intensity=(intt, intt+0.2))(data))[0, :, :, :]
                np.save(os.path.join(dataset_path_dist, level, filename), data_dist)


def random_spike_distortion(ds_type_combinations):
    # add random spike artifacts
    print('coming into function: random_spike_distortion')

    for ds_name, sub_type in ds_type_combinations:
        print(ds_name, sub_type)

        dataset_path_src = f"./datasets/src/{ds_name}_{sub_type}/"
        dataset_path_dist = f"./datasets/dist/{ds_name}_{sub_type}/"
        path_list = os.listdir(dataset_path_src)
        path_list.sort()

        
        nums = [2, 4, 7, 7]
        ints = [0.8, 1.0, 1.2, 2]
        
        levels = ['spike_1', 'spike_2', 'spike_3', 'spike_4']

        for level in levels:
            if not os.path.exists(os.path.join(dataset_path_dist, level)):
                os.mkdir(os.path.join(dataset_path_dist, level))

        count = 0
        for filename in path_list:
            data = np.load(os.path.join(dataset_path_src, filename))
            data = data.reshape(1, *data.shape)

            count += 1
            if count % 10 == 0:
                print('\t', ds_name, sub_type, count)

            for num, intt, level in zip(nums, ints, levels):
                data_dist = np.array(RandomSpike(num_spikes=(num-1, num+1), intensity=(intt, intt+0.2))(data))[0, :, :, :]
                np.save(os.path.join(dataset_path_dist, level, filename), data_dist)


def random_noise_distortion(ds_type_combinations):
    print('coming into function: random_noise_distortion')
    # add random noise artifacts
    for ds_name, sub_type in ds_type_combinations:
        print(ds_name, sub_type)

        dataset_path_src = f"./datasets/src/{ds_name}_{sub_type}/"
        dataset_path_dist = f"./datasets/dist/{ds_name}_{sub_type}/"
        path_list = os.listdir(dataset_path_src)
        path_list.sort()

        
        means = [0.1, 0.4, 0.8, 10]
        stds = [100.0, 400.0, 800.0, 3000.0]
        
        levels = ['noise_1', 'noise_2', 'noise_3', 'noise_4']

        for level in levels:
            if not os.path.exists(os.path.join(dataset_path_dist, level)):
                os.mkdir(os.path.join(dataset_path_dist, level))

        count = 0
        for filename in path_list:
            data = np.load(os.path.join(dataset_path_src, filename))
            data = data.reshape(1, *data.shape)

            count += 1
            if count % 10 == 0:
                print('\t', ds_name, sub_type, count)

            for mean, std, level in zip(means, stds, levels):
                data_dist = np.array(RandomNoise(mean=0, std=(std, std+0.1))(data))[0, :, :, :]
                np.save(os.path.join(dataset_path_dist, level, filename), data_dist)


def random_blur_distortion(ds_type_combinations):
    print('coming into function: random_blur_distortion')
    # add random blur artifacts
    for ds_name, sub_type in ds_type_combinations:
        print(ds_name, sub_type)

        dataset_path_src = f"./datasets/src/{ds_name}_{sub_type}/"
        dataset_path_dist = f"./datasets/dist/{ds_name}_{sub_type}/"
        path_list = os.listdir(dataset_path_src)
        path_list.sort()

        stds = [0.8, 1.2, 2.5, 5]
        
        levels = ['blur_1', 'blur_2', 'blur_3', 'blur_4']

        for level in levels:
            if not os.path.exists(os.path.join(dataset_path_dist, level)):
                os.mkdir(os.path.join(dataset_path_dist, level))

        count = 0
        for filename in path_list:
            data = np.load(os.path.join(dataset_path_src, filename))
            plt.imshow(data[:, :, data.shape[-1] // 2], cmap='gray')
            plt.show()

            data = data.reshape(1, *data.shape)

            count += 1
            if count % 10 == 0:
                print('\t', ds_name, sub_type, count)

            for std, level in zip(stds, levels):
                data_dist = np.array(RandomBlur(std=(std, std+0.1, std, std+0.1, std, std+0.1))(data))[0, :, :, :]
                np.save(os.path.join(dataset_path_dist, level, filename), data_dist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--dist', type=str, default='motionm', help='distortion (artifacts) type')
    args = parser.parse_args()

    seed_everything(seed=42)
    make_dirs()

    ds_type_combinations = [('OASIS', 'T1w'), ('NFBS', 'T1w'), ('IXI', 'T1w'), ('IXI', 'T2w'), ('QTAB', 'T1w'), ('QTAB', 'T2w'), ('ARC', 'T1w'), ('ARC', 'T2w'), ('ABIDE', 'ccs'), ('ABIDE', 'cpac'), ('ABIDE', 'dparsf'), ('ABIDE', 'niak')]

    print(args.dist, type(args.dist))
    
    if args.dist == 'motion':
        random_motion_distortion(ds_type_combinations)
    elif args.dist == 'ghosting':
        random_ghosting_distortion(ds_type_combinations)
    elif args.dist == 'spike':
        random_spike_distortion(ds_type_combinations)
    elif args.dist == 'noise':
        random_noise_distortion(ds_type_combinations)
    elif args.dist == 'blur':
        random_blur_distortion(ds_type_combinations)
    else:
        raise ValueError(f"The argument dist ({args.dist}) should be one of ['motion', 'ghosting', 'spike', 'noise', 'blur'].")
