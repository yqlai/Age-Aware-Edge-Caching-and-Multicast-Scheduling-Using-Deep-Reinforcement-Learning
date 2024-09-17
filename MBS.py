from SBS import SBS
import torch
import torch.nn as nn
import torch.optim as optim

class MBS:
    def __init__(self):
        s
    
    def decide(self, sbs, id, reward):
        min_id = sbs.cache[0].id
        min_freq = 100
        for i in range(sbs.cache_size):
            if sbs.cache[i].used < min_freq:
                min_freq = sbs.cache[i].used
                min_id = i
        
        return sbs.cache[min_id].id

        