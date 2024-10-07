from SBS import SBS
import torch
import torch.nn as nn
import torch.optim as optim

class MBS:
    def __init__(self):
        pass
    
    def decide(self, sbs, id, reward, method='MA'):
        
        ma_id = self.MA(sbs)
        lru_id = self.LRU(sbs)
        if method == 'MA':
            return ma_id
        elif method == 'LRU':
            return lru_id
        else:
            return -1

    def MA(self, sbs):
        ma_id = 0
        highest_age = sbs.cache[0].age
        for i in range(sbs.cache_size):
            if sbs.cache[i].age > highest_age:
                highest_age = sbs.cache[i].age
                ma_id = i
        return sbs.cache[ma_id].id

    def LRU(self, sbs):
        lru_id = 0
        recent_time_slot = sbs.cache[0].recent_time_slot
        for i in range(sbs.cache_size):
            if sbs.cache[i].recent_time_slot < recent_time_slot:
                recent_time_slot = sbs.cache[i].recent_time_slot
                lru_id = i
        return sbs.cache[lru_id].id