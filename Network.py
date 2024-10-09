import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class Network(nn.Module):
    def __init__(self, input_size, output_size, hiddden_size=60):
        super(Network, self).__init__()
        self.fc = nn.Linear(input_size, hiddden_size)
        self.out = nn.Linear(hiddden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc(x))
        x = self.out(x)
        x = F.softmax(x, dim=1)
        return x

# states(k) include: [{q}, A, C, j](k) and r(k-1)
# {q}: the queue length of the user request queue       size: N (num_content)
# A: the age of the content in the cache                size: M (cache_size)
# C: the indices of cached contents                     size: M (cache_size)
# j: the index of the content to be updated             size: 1
# r: the reward of the previous epoch                   size: 1

# actions: the index of the content to be updated       size: 1
# action_space: [0, N]

class DQN(object):
    def __init__(self, num_content, cache_size):
        self.state_size = num_content + 2*(cache_size+1)
        self.action_space = cache_size + 1
        self.eval_net = Network(self.state_size, self.action_space)
        self.target_net = Network(self.state_size, self.action_space)
        
        # For updating the target network
        self.learn_step_counter = 0
        self.learn_step = 50

        self.memory_counter = 0
        self.memory_capacity = 10000
        self.memory = np.zeros((self.memory_capacity, self.state_size*2+2)) # store s, a, r, s_

        self.batch_size = 50
        self.epsilon = 0.9
        self.epsilon_decrease_rate = 0.001
        self.learning_rate = 0.001
        self.beta = 0.95 # discount factor

        self.optimzer = optim.Adam(self.eval_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
    

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < self.epsilon:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy().data[0]
        else:
            action = np.random.randint(0, self.action_space)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_capacity
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def learn(self):
        if self.memory_counter < self.batch_size:
            return
        
        if self.learn_step_counter % self.learn_step == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(min(self.memory_counter, self.memory_capacity), self.batch_size, replace=False)
        b_memory = self.memory[sample_index, :]

        b_s = torch.FloatTensor(b_memory[:, :self.state_size])
        b_a = torch.LongTensor(b_memory[:, self.state_size:self.state_size+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.state_size+1:self.state_size+2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.state_size:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.beta * q_next.max(1)[0].view(self.batch_size, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimzer.zero_grad()
        loss.backward()
        self.optimzer.step()
