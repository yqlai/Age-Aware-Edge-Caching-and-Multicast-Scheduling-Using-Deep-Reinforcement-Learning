from Queues import User_Request_Queue
from SBS import SBS, Decision_Making
from MBS import MBS
from utils import plot_AAoI
from train import train

import torch
import torch.nn as nn

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f'Using device: {device}')

    method_flag = [0, 0, 0, 1]

    num_content = 30 # N
    cache_size = 5   # M
    arr_aoi_MA, arr_aoi_LRU, arr_aoi_LFU, arr_aoi_RL = [], [], [], []
    arr_user_request_MA, arr_user_request_LRU, arr_user_request_LFU, arr_user_request_RL = [], [], [], []
    
    sbs = SBS(cache_size, num_content)
    mbs = MBS(num_content, cache_size)

    num_epochs = 10000

    if method_flag[0]:
        mbs.initialize()
        sbs.initialize()
        reward = 0
        time_slot = 0
        arr_aoi_MA, arr_user_request_MA = Decision_Making(mbs, sbs, num_epochs, method='MA')

    if method_flag[1]:
        mbs.initialize()
        sbs.initialize()
        reward = 0
        time_slot = 0
        arr_aoi_LRU, arr_user_request_LRU = Decision_Making(mbs, sbs, num_epochs, method='LRU')

    if method_flag[2]:
        mbs.initialize()
        sbs.initialize()
        reward = 0
        time_slot = 0
        arr_aoi_LFU = Decision_Making(mbs, sbs, num_epochs, method='LFU')
        arr_aoi_LFU, arr_user_request_LFU = Decision_Making(mbs, sbs, num_epochs, method='LFU')

    if method_flag[3]:
        mbs.initialize()
        sbs.initialize()
        reward = 0
        time_slot = 0
        arr_aoi_RL, arr_user_request_RL = train(mbs, sbs, num_epochs)

    # Only plot the arr that with True flag
    plot_arrs, plot_labels = [], []
    plot_data = [
        (arr_aoi_MA, 'MA'),
        (arr_aoi_LRU, 'LRU'),
        (arr_aoi_LFU, 'LFU'),
        (arr_aoi_RL, 'RL')
    ]

    for flag, (arr, label) in zip(method_flag, plot_data):
        if flag:
            plot_arrs.append(arr)
            plot_labels.append(label)

    plot_AAoI(plot_arrs, num_epochs, window=100, labels=plot_labels, save=False)
    
    
    plot_arrs, plot_labels = [], []
    plot_data = [
        (arr_user_request_MA, 'MA'),
        (arr_user_request_LRU, 'LRU'),
        (arr_user_request_LFU, 'LFU'),
        (arr_user_request_RL, 'RL')
    ]

    for flag, (arr, label) in zip(method_flag, plot_data):
        if flag:
            plot_arrs.append(arr)
            plot_labels.append(label)
        
    plot_AAoI(plot_arrs, num_epochs, window=100, labels=plot_labels, save=False)

    print('End of Simulation')