import torch
import torch.nn as nn
import numpy
import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return(
            torch.tensor(numpy.array(states), dtype=torch.float32),
            torch.tensor(numpy.array(actions), dtype=torch.long),
            torch.tensor(numpy.array(rewards), dtype=torch.float32),
            torch.tensor(numpy.array(next_states), dtype=torch.float32),
            torch.tensor(numpy.array(dones), dtype=torch.float32)
        )
    
    def __len__(self):
        return len(self.buffer)
    
class DQNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.layers(x)
    
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.action_size = action_size
        self.epsilon = 1.0
        self.epsilon_decay = 0.9986
        self.epsilon_min = 0.01
        self.gamma = 0.9
        self.lr = 0.001
        self.batch_size = 64

        self.memory = ReplayBuffer()
        self.network = DQNetwork(state_size, action_size)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            q_values = self.network(state_tensor)
            return torch.argmax(q_values).item()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # what the network currently predicts
        current_q = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # what is should have predicted - Bellman equation
        with torch.no_grad():
            next_q = self.network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # loss and backpropegation
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()