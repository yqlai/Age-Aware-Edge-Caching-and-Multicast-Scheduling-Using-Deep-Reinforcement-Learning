import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from Network import Network, DQN
from SBS import SBS
from MBS import MBS

# states(k) include: [{q}, A, C, j](k) and r(k-1)
# {q}: the queue length of the user request queue       size: N (num_content)
# A: the age of the content in the cache                size: M (cache_size)
# C: the indices of cached contents                     size: M (cache_size)
# j: the index of the content to be updated             size: 1
# r: the reward of the previous epoch                   size: 1

# actions: the index of the content to be updated       size: 1
# action_space: [0, N]

def train(mbs, sbs, num_epoch):
    update_id = 10
    reward = 0
    arr_aoi_ages = []
    arr_aoi_requests = []
    sum_arr_aoi_ages = 1
    sum_arr_aoi_requests = 1
    arr_aoi = []

    arr_actions = []

    previous_state = np.zeros(sbs.num_content + 2*(sbs.cache_size+1))
    current_state = sbs.user_request.queue + [c.age for c in sbs.cache] + [c.id for c in sbs.cache] + [update_id, reward]


    time_slot = 0
    time_slot_k = 0
    epoch = 0

    # pbar = tqdm(total=num_epoch)
    while epoch < num_epoch:
        # tqdm.update(pbar, epoch)
        
        # Get the update_id, replace_id from MBS using DQN model
        if update_id in sbs.cache:
            replace_id = sbs.cache.index(update_id)
            arr_actions.append(replace_id)
        else:
            replace_id = mbs.decide(sbs, current_state, method='RL')
            arr_actions.append(replace_id)
            replace_id -=1
            if replace_id >= 0:
                replace_id = sbs.cache[replace_id].id
            else:
                replace_id = -1
        

        # print(f'Arr_actions: {arr_actions}')

        # Replay Buffer needs [s(k), a(k), r(k), s]
        # Current state         -> s(k)
        # Previous state        -> s(k-1)
        # replace_id            -> a(k)
        # reward                -> r(k-1)
        # arr_actions[-2]       -> a(k-1)
        
        previous_state = current_state

        if len(arr_actions) > 5:
            mbs.agent.store_transition(previous_state, arr_actions[-2], reward, current_state)
        mbs.agent.learn()
        
        if not sbs.replace(update_id, replace_id, time_slot):
            print('Some problems occur...')
        
        mu = update_id
        alpha = 1
        reward = 0

        mu_age = 1
        if not (replace_id == -1):
            mu_age = sbs.cache[sbs.cache.index(mu)].age
        arr_aoi_ages.append(sbs.user_request.queue[mu] * mu_age)
        arr_aoi_requests.append(sbs.user_request.queue[mu])
        sum_arr_aoi_ages += sbs.user_request.queue[mu] * mu_age
        sum_arr_aoi_requests += sbs.user_request.queue[mu]
        arr_aoi.append(sum_arr_aoi_ages / sum_arr_aoi_requests)

        sbs.user_request.service(mu)
        if not replace_id == -1:
            sbs.cache[sbs.cache.index(mu)].used.append(time_slot)
            sbs.cache[sbs.cache.index(mu)].recent_time_slot = time_slot

        first_time_slot_of_epoch = 1
        while epoch < num_epoch:
            if epoch % 100 == 0:
                print(f'Epoch: {epoch}, Time slot: {time_slot}')
            # Environment step
            if alpha == 1 and (not (replace_id == -1)):
                sbs.cache[sbs.cache.index(update_id)].age = 0
            sbs.step()

            previous_state = current_state
            current_state = sbs.user_request.queue + [c.age for c in sbs.cache] + [c.id for c in sbs.cache] + [update_id, reward]

            age_t = 1
            if first_time_slot_of_epoch and (not (replace_id == -1)):
                age_t = sbs.cache[sbs.cache.index(update_id)].age
                first_time_slot_of_epoch = 0
            reward = reward + sbs.user_request.queue[mu] * (age_t * (1 - alpha) + 1)
            # print('----------------------------------------------------------------')
            # print(f'Epoch: {epoch:4}, Time slot: {time_slot:4}')
            # print(f'id: [{sbs.cache[0].id}, {sbs.cache[1].id}, {sbs.cache[2].id}, {sbs.cache[3].id}, {sbs.cache[4].id}]')
            # print(f'Age: [{sbs.cache[0].age}, {sbs.cache[1].age}, {sbs.cache[2].age}, {sbs.cache[3].age}, {sbs.cache[4].age}]')
            # print(f'Used: [{sbs.cache[0].used}, {sbs.cache[1].used}, {sbs.cache[2].used}, {sbs.cache[3].used}, {sbs.cache[4].used}]')
            # print(f'Reward: {reward}')
            time_slot += 1
            
            mu, alpha = sbs.decide()
            # print(f'mu: {mu}, alpha: {alpha}')
            # print('----------------------------------------------------------------')
            
            # Update the cache from MBS
            if alpha == 1:
                time_slot_k = time_slot
                epoch += 1
                update_id = mu
                break
            # Multicast the cache content
            else:
                arr_aoi_ages.append(sbs.user_request.queue[mu] * sbs.cache[sbs.cache.index(mu)].age)
                arr_aoi_requests.append(sbs.user_request.queue[mu])
                sum_arr_aoi_ages += sbs.user_request.queue[mu] * sbs.cache[sbs.cache.index(mu)].age
                sum_arr_aoi_requests += sbs.user_request.queue[mu]
                arr_aoi.append(sum_arr_aoi_ages / sum_arr_aoi_requests)
                sbs.user_request.service(mu)
                sbs.cache[sbs.cache.index(mu)].used.append(time_slot)
    return arr_aoi
