from Queues import User_Request_Queue

class Cached_Content:
    def __init__(self, id, age=0):
        self.id = id
        self.age = age
        self.used = 0
    
    def __eq__(self, other:int):
        if self.id == other:
            return True
        return False
    
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

    def replace(self, update_id, remove_id):
        for i in range(self.cache_size):
            if self.cache[i].id == remove_id:
                self.cache[i] = Cached_Content(update_id, 0)
                return 1
        return 0
    
    def step(self):
        self.user_request.step()
        for content in self.cache:
            content.step()
        
        self.pseudo_queue_u = max(self.pseudo_queue_u - self.D, 0) + self.last_alpha
        self.pseudo_queues = [max(self.pseudo_queues[i] - self.user_request.arr_prob[i], 0) + self.user_request.queue[i] for i in range(self.num_content)]

        for i in range(self.cache_size):
            self.cache[i].step()

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
        print(f'Update: {update}, Not Update: {not_update}')
        
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

def Decision_Making(mbs, sbs, num_time_slots):
    update_id = 0
    reward = 0

    time_slot = 0
    epoch = 0
    while time_slot < num_time_slots:
        replace_id = mbs.decide(sbs, update_id, reward)
        print('----------------------------------------------------------------')
        print(f'Epoch: {epoch:3}, Update id: {update_id:3}, Replace id: {replace_id:3}')

        if sbs.replace(update_id, replace_id):
            print("Cached Content: ", [content.id for content in sbs.cache])
        else:
            print('Some problems occur...')

        mu = update_id
        alpha = 1
        reward = 0

        sbs.user_request.service(mu)
        sbs.cache[sbs.cache.index(mu)].used += 1

        while time_slot < num_time_slots:
            if alpha == 1:
                sbs.cache[sbs.cache.index(update_id)].age = 0
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
                sbs.user_request.service(mu)
                sbs.cache[sbs.cache.index(mu)].used += 1