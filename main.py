from Queues import User_Request_Queue
from SBS import SBS, Decision_Making
from MBS import MBS
from utils import plot_AAoI

import torch
import torch.nn as nn

if __name__ == '__main__':
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f'Using device: {device}')

    num_content = 30 # N
    cache_size = 5   # M
    
    sbs = SBS(cache_size, num_content)
    mbs = MBS()

    sbs.initialize()

    reward = 0

    num_time_slots = 10000
    num_epochs = 10000
    time_slot = 0
    
    arr_aoi_MA = Decision_Making(mbs, sbs, num_epochs, method='MA')

    sbs.initialize()
    reward = 0
    time_slot = 0
    arr_aoi_LRU = Decision_Making(mbs, sbs, num_epochs, method='LRU')

    sbs.initialize()
    reward = 0
    time_slot = 0
    arr_aoi_LFU = Decision_Making(mbs, sbs, num_epochs, method='LFU')

    plot_AAoI([arr_aoi_MA, arr_aoi_LRU, arr_aoi_LFU], num_time_slots, window=100, labels=['MA', 'LRU', 'LFU'])

    print('End of Simulation')