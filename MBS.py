from SBS import SBS
import torch
import torch.nn as nn
import torch.optim as optim

class MBS:
    def __init__(self):
        pass
    
    def decide(self, sbs, id, reward, time_slot=0, method='MA'):
        
        if method == 'MA':
            return self.MA(sbs)
        elif method == 'LRU':
            return self.LRU(sbs)
        elif method == 'LFU':
            return self.LFU(sbs)
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
    
    def LFU(self, sbs):
        arr_freq = [len(sbs.cache[i].used) for i in range(sbs.cache_size)]
        min_freq = min(arr_freq)

        arr_min_freq_id = [i for i in range(sbs.cache_size) if len(sbs.cache[i].used) == min_freq]

        lfu_id = arr_min_freq_id[0]
        least_order = sbs.cache[arr_min_freq_id[0]].order
        for i in range(len(arr_min_freq_id)):
            if sbs.cache[arr_min_freq_id[i]].order < least_order:
                lfu_id = arr_min_freq_id[i]
                least_order = sbs.cache[arr_min_freq_id[i]].order
        return sbs.cache[lfu_id].id
