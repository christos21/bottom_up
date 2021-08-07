import os
import pathlib
from scipy.io import loadmat, savemat
import pandas as pd
import numpy as np
import re


CURRENT_PATH = pathlib.Path(__file__).parent.absolute()


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def from_csv_to_mat():
    """
    Transform the directory of load profiles to a .mat file.
    This is needed only for sharing/uploading the profiles since .mat files are gzip compressed HDF5.
    :return:
    """
    # get path to profiles directory
    profiles_path = os.path.join(CURRENT_PATH, 'profiles')
    # get names of appliance folders
    appliance_folders = os.listdir(profiles_path)
    # create an empty dict to store arrays
    data_dict = dict()
    # for each available appliance
    for appliance in appliance_folders:

        if appliance in ['PV']:
            appliance_profile_path = os.path.join(profiles_path, appliance, 'monthly_profiles.csv')
            df = pd.read_csv(appliance_profile_path, index_col=0)
            P = df.values
            data_dict[appliance] = P
            continue

        for day_type in ['WD', 'NWD']:
            appliance_profile_path = os.path.join(profiles_path, appliance, day_type)
            profiles = os.listdir(appliance_profile_path)
            profiles.sort(key=natural_keys)

            P = np.zeros([24 * 60 * 60, len(profiles)])
            Q = np.zeros([24 * 60 * 60, len(profiles)])

            # keep flags in case there is no P or Q
            p_flag = False
            q_flag = False

            for i, profile in enumerate(profiles):
                df = pd.read_csv(os.path.join(appliance_profile_path, profile), index_col=0)

                if 'P' in df.columns:
                    P[:, i] = df.P.values.reshape(-1)
                    p_flag = True
                if 'Q' in df.columns:
                    Q[:, i] = df.Q.values.reshape(-1)
                    q_flag = True

            if p_flag:
                key = appliance + '_P_' + day_type
                data_dict[key] = P
            if q_flag:
                key = appliance + '_Q_' + day_type
                data_dict[key] = Q

    savemat(os.path.join(CURRENT_PATH, 'profiles.mat'), data_dict, do_compression=True)


def from_mat_to_csv(mat_file=os.path.join(CURRENT_PATH, 'profiles.mat')):
    """
    Transform the mat file of load profiles to a directory tree with csv for each profile.
    A base directory is created named 'profiles'. Inside 'profiles' there are folders for each available appliance.
    In each appliance folder there are two folders for working-days 'WD' and non-working-days 'NWD' for compatibility
    with the current implementation. Finally, inside the last folder there are CSV files for each profile containing
    active and reactive power if available.
    In this way, it is vary easy to add new profiles and new appliances.
    :param mat_file: str, path to the initial .mat file
    :return:
    """
    profiles_csv_path = os.path.join(CURRENT_PATH, 'profiles')
    if not os.path.exists(profiles_csv_path):
        os.makedirs(profiles_csv_path)

    mat_file = loadmat(mat_file)

    keys = set(mat_file.keys()) - {'__header__', '__version__', '__globals__'}

    for key in keys:
        if key == 'PV':
            directory = os.path.join(profiles_csv_path, key)
            if os.path.exists(directory):
                continue
            else:
                os.makedirs(directory)

            P = mat_file[key]
            df = pd.DataFrame(data=P, columns=['January', 'February', 'March', 'April', 'May',
                                               'June', 'July', 'August', 'September', 'October',
                                               'November', 'December'])
            df.to_csv(os.path.join(directory, 'monthly_profiles.csv'))

        else:
            appliance, power_type, day = key.split('_')
            directory = os.path.join(profiles_csv_path, appliance, day)
            if os.path.exists(directory):
                continue
            else:
                os.makedirs(directory)

            power_types = [k.split('_')[1] for k in keys if appliance in k]

            q_flag = True if 'Q' in power_types else False

            P = mat_file[appliance + '_P_' + day]

            if q_flag:
                Q = mat_file[appliance + '_Q_' + day]

            number_of_profiles = P.shape[1]

            for i in range(number_of_profiles):
                if q_flag:
                    df = pd.DataFrame(data=np.hstack([P[:, i].reshape(-1, 1), Q[:, i].reshape(-1, 1)]),
                                      columns=['P', 'Q'])
                else:
                    df = pd.DataFrame(data=P[:, i], columns=['P'])

                df.to_csv(os.path.join(directory, 'profile_' + str(i) + '.csv'))



