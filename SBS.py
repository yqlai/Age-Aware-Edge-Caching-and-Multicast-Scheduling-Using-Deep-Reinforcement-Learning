from Queues import User_Request_Queue
from utils import plot_AAoI
from tqdm import tqdm

class Cached_Content:
    def __init__(self, id, age=0, order=0):
        self.id = id
        self.age = age
        self.used = [] # Record the time slot when the content is used
        self.recent_time_slot = 0 # Record the recent time slot when the content is used
        
        self.order = order
        self.LFU_window = 10
    
    def __eq__(self, other:int):
        if self.id == other:
            return True
        return False
    
    def LFU_update(self, t):
        if self.age >= 10:
            self.used = []
            return
        for i in range(len(self.used)):
            if self.used[i] < t:
                self.used.pop(i)
                break
    
    def step(self):
        self.age += 1

class SBS:
    def __init__(self, cache_size, num_content):
        self.cache_size = cache_size
        self.num_content = num_content

        self.cache = [Cached_Content(i, 0) for i in range(cache_size)]
        self.user_request = User_Request_Queue(num_content)

        self.pseudo_queue_u = 0
        self.pseudo_queues = [0 for i in range(num_content)]

        self.last_alpha = 0
        self.D = 0.3
        self.V = 0.5
        
    
    def initialize(self):
        self.cache = [Cached_Content(i, 0) for i in range(self.cache_size)]
        self.user_request.initialize()

        self.pseudo_queue_u = 0
        self.pseudo_queues = [0 for i in range(self.num_content)]

    def replace(self, update_id, remove_id, time_slot):
        if remove_id == -1:
            return 1
        for i in range(self.cache_size):
            if self.cache[i].id == remove_id:
                self.cache[i] = Cached_Content(update_id, 0, time_slot)
                return 1
        return 0
    
    def step(self):
        self.user_request.step()
        for content in self.cache:
            content.step()
        
        self.pseudo_queue_u = max(self.pseudo_queue_u - self.D, 0) + self.last_alpha
        self.pseudo_queues = [max(self.pseudo_queues[i] - self.user_request.arr_prob[i], 0) + self.user_request.queue[i] for i in range(self.num_content)]

    def decide(self):
        # Grouping the contents to be updated and not to be updated
        update = []
        not_update = []
        for i in range(self.num_content):
            if not i in self.cache:
                update.append(i)
            else:
                if self.pseudo_queue_u < self.V * self.user_request.queue[i] * self.cache[self.cache.index(i)].age:
                    update.append(i)
                else:
                    not_update.append(i)
        
        # print the contents to be updated and not to be updated
        # print(f'Update: {update}, Not Update: {not_update}')
        
        l = -1
        min_value = 100
        for i in update:
            tmp = self.user_request.queue[i] * (self.V - self.pseudo_queues[i])
            if tmp < min_value:
                min_value = tmp
                l = i
        m = -1
        min_value = 100
        for i in not_update:
            tmp = self.user_request.queue[i] * (self.V * (1 + self.cache[self.cache.index(i)].age) - self.pseudo_queues[i])
            if tmp < min_value:
                min_value = tmp
                m = i
        
        if l == -1:
            return m, 0
        if m == -1:
            return l, 1
        
        left_side = self.pseudo_queue_u
        right_side = self.user_request.queue[m] * (self.V * (1 + self.cache[self.cache.index(m)].age) - self.pseudo_queues[m]) - self.user_request.queue[l] * (self.V - self.pseudo_queues[l])
        if left_side < right_side:
            mu = l
            alpha = 1
        else:
            mu = m
            alpha = 0

        return mu, alpha

def Decision_Making(mbs, sbs, num_epochs, method='MA'):
    update_id = 0
    reward = 0
    arr_aoi_ages = []
    arr_aoi_requests = []
    sum_arr_aoi_ages = 1
    sum_arr_aoi_requests = 1 # to avoid zero division
    arr_aoi = []

    time_slot = 0
    epoch = 0

    pbar = tqdm(total=num_epochs)
    while epoch < num_epochs:
        tqdm.update(pbar, epoch)
        replace_id = mbs.decide(sbs, None, method=method)
        # print('----------------------------------------------------------------')
        # print(f'Epoch: {epoch:3}, Update id: {update_id:3}, Replace id: {replace_id:3}')

        if sbs.replace(update_id, replace_id, time_slot):
            pass
            # print(f'Update_id: {update_id} is replaced by {replace_id}')
            # print(f'Age of update_id: {sbs.cache[sbs.cache.index(update_id)].age}')
        else:
            print('Some problems occur...')

        mu = update_id
        alpha = 1
        reward = 0

        arr_aoi_ages.append(sbs.user_request.queue[mu] * sbs.cache[sbs.cache.index(mu)].age)
        arr_aoi_requests.append(sbs.user_request.queue[mu])
        sum_arr_aoi_ages += sbs.user_request.queue[mu] * sbs.cache[sbs.cache.index(mu)].age
        sum_arr_aoi_requests += sbs.user_request.queue[mu]
        arr_aoi.append(sum_arr_aoi_ages / sum_arr_aoi_requests)
        # if method == 'LFU':
        #     print('----------------------------------------------------------------')
        #     print(f'Age: [{sbs.cache[0].age}, {sbs.cache[1].age}, {sbs.cache[2].age}, {sbs.cache[3].age}, {sbs.cache[4].age}]')
        #     print(f'Used: [{sbs.cache[0].used}, {sbs.cache[1].used}, {sbs.cache[2].used}, {sbs.cache[3].used}, {sbs.cache[4].used}]')
        sbs.user_request.service(mu)
        sbs.cache[sbs.cache.index(mu)].used.append(time_slot)
        sbs.cache[sbs.cache.index(mu)].recent_time_slot = time_slot

        for cache in sbs.cache:
            cache.LFU_update(time_slot-cache.LFU_window)

        while epoch < num_epochs:
            if alpha == 1:
                sbs.cache[sbs.cache.index(update_id)].age = 1
            sbs.step()

            reward = reward + sbs.user_request.queue[mu] * (sbs.cache[sbs.cache.index(update_id)].id * (1 - alpha) + 1)
            time_slot += 1

            mu, alpha = sbs.decide()

            if not mu in sbs.cache:
                time_slot_k = time_slot
                epoch += 1
                update_id = mu
                break
            else:
                arr_aoi_ages.append(sbs.user_request.queue[mu] * sbs.cache[sbs.cache.index(mu)].age)
                arr_aoi_requests.append(sbs.user_request.queue[mu])
                sum_arr_aoi_ages += sbs.user_request.queue[mu] * sbs.cache[sbs.cache.index(mu)].age
                sum_arr_aoi_requests += sbs.user_request.queue[mu]
                arr_aoi.append(sum_arr_aoi_ages / sum_arr_aoi_requests)
                sbs.user_request.service(mu)
                sbs.cache[sbs.cache.index(mu)].used.append(time_slot)
    # plot_AAoI(arr_aoi, time_slot, window=800)
    return arr_aoi