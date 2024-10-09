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

    num_content = 30 # N
    cache_size = 5   # M
    
    sbs = SBS(cache_size, num_content)
    mbs = MBS(num_content, cache_size)

    mbs.initialize()
    sbs.initialize()

    num_epochs = 10000
    # train(mbs, sbs, num_epochs)

    mbs.initialize()
    sbs.initialize()
    reward = 0
    time_slot = 0
    arr_aoi_MA = Decision_Making(mbs, sbs, num_epochs, method='MA')

    mbs.initialize()
    sbs.initialize()
    reward = 0
    time_slot = 0
    arr_aoi_LRU = Decision_Making(mbs, sbs, num_epochs, method='LRU')

    mbs.initialize()
    sbs.initialize()
    reward = 0
    time_slot = 0
    arr_aoi_LFU = Decision_Making(mbs, sbs, num_epochs, method='LFU')

    # mbs.initialize()
    # sbs.initialize()
    # reward = 0
    # time_slot = 0
    # arr_aoi_RL = train(mbs, sbs, num_epochs)

    plot_AAoI([arr_aoi_MA, arr_aoi_LRU, arr_aoi_LFU], num_epochs, window=100, labels=['MA', 'LRU', 'LFU'])

    print('End of Simulation')