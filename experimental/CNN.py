import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

from torchvision.transforms import v2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from IPython import display
# 'rgb_array'
env = gym.make('ALE/Breakout-v5', render_mode='human', obs_type='rgb', )
env.metadata['render_fps'] = 60

env.reset()



plt.ion()

device = "cuda"

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    def __init__(self, input_width, input_height, n_actions):
        super(DQN, self).__init__()
        self.transform = v2.Compose([
            v2.Resize((84, 84)),
            v2.ToTensor()
        ])
        self.conv1 = nn.Conv2d(1, 64, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=4, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
    
    def forward(self, x):
        x = self.transform(x)
        # breakpoint()
        x = x.transpose(1, 0)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
        
        # tmp = x.transpose(1, 3)
        
        # tmp = F.relu(self.conv1(tmp))
        # tmp = F.relu(self.conv2(tmp))
        # tmp = x.view(tmp.size(0), -1)
        # tmp = F.relu(self.fc1(tmp))
        # return self.fc2(tmp)

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.99
EPS_END = 0.05
# EPS_DECAY = 1e5
EPS_DECAY = 100
TAU = 0.005
LR = 1e-4

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(state.shape[0], state.shape[1], n_actions).to(device)
target_net = DQN(state.shape[0], state.shape[1], n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.SGD(policy_net.parameters(), lr=LR, momentum=0.9)
memory = ReplayMemory(10000)

steps_done = 0
episodes_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # Exploit
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        # Explore
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
scores = []

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(scores, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(durations_t.numpy())
    
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    
    plt.pause(0.001)
    if not show_result:
        display.display(plt.gcf())
        display.clear_output(wait=True)
    else:
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    
    criterion = nn.CrossEntropyLoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

num_episodes = 500
current_reward = 0
print(env.metadata['render_fps'])
for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    env.render()

    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        if reward > 0: current_reward += reward
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            scores.append(current_reward)
            current_reward = 0
            episodes_done += 1
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
            print("eps_threshold", eps_threshold)
            # policy_net.save("model.pth")
            plot_durations()
            break
torch.save(policy_net.state_dict(), "model.pth")
print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()