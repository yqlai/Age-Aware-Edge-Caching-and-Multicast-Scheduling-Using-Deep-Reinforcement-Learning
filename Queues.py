import numpy as np
from utils import Zipf_dist

class User_Request_Queue:
    def __init__(self, num_content):
        self.num_content = num_content
        self.queue = [0 for i in range(num_content)]

        self.arr_prob = Zipf_dist(num_content, 1)
        self.arr_content_per_slot = 10

        self.capacity = 5000

    def initialize(self):
        self.queue = [0 for i in range(self.num_content)]

    def service(self, content_id):
        self.queue[content_id] = 0
    
    def step(self):
        arrival_contents = np.random.choice(range(self.num_content), size=self.arr_content_per_slot, p=self.arr_prob)
        arrival_counts = np.bincount(arrival_contents, minlength=self.num_content)
        self.queue = np.minimum(self.queue + arrival_counts, self.capacity).tolist()