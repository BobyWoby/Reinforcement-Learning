import gymnasium as gym
from gymnasium.wrappers import frame_stack, atari_preprocessing
import math
import random
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import keyboard
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from IPython import display
import torchvision.transforms as T




env = gym.make('ALE/Breakout-v5', render_mode='human').unwrapped
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
        print(input_width, input_height, n_actions)
        self.flatten = nn.Flatten()
        n_observations = input_width * input_height
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer3 = nn.Linear(128, n_actions)
    
    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.layer1(x))
        return self.layer3(x)

class DQN2(nn.Module):
    def __init__(self, n_actions):
        super(DQN2, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=8, stride=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=8, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = nn.Linear(16 * 11 * 8, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
    
    def forward(self, x):
        x = x[:, None]
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 11 * 8)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x) 

class DQN3(nn.Module):
    def __init__(self, n_actions, dropout=0.2, init_weights=True):
        
        super(DQN3, self).__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(True)
                                        )
        self.classifier = nn.Sequential(nn.Linear(64*22*16, 64),
                                        nn.ReLU(True),
                                        nn.Linear(64, n_actions)
                                        )
        self.cnn1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.cnn2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.cnn3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.classifier1 = nn.Linear(64*22*16, 64)
        self.classifier2 = nn.Linear(64, n_actions)
        if init_weights:
            self._initialize_weights()
    def forward(self, x):
        
        # breakpoint()
        
        x = F.relu(self.cnn1(x))
        x = F.relu(self.cnn2(x))
        x = F.relu(self.cnn3(x))
        
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.classifier1(x))
        
        x = self.classifier2(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

class Dueling_DQN_2016_Modified(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super().__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4,bias=False),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2,bias=False),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1,bias=False),
                                        nn.ReLU(True),
                                        nn.Conv2d(64,1024,kernel_size=7,stride=1,bias=False),
                                        nn.ReLU(True)
                                        )
        self.streamA = nn.Linear(81920, num_classes)
        self.streamV = nn.Linear(81920, 1)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # breakpoint()
        x = self.cnn(x)
        sA,sV = torch.split(x,512,dim = 0)
        sA = torch.flatten(sA, start_dim=0)
        sV = torch.flatten(sV, start_dim=0)
        sA = self.streamA(sA) #(B,4)
        sV = self.streamV(sV) #(B,1)
        # combine this 2 values together
        # breakpoint()
        Q_value = sV + (sA - torch.mean(sA,dim=0,keepdim=True))
        return Q_value #(B,4)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)


# Hyperparameters
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 1e5
# EPS_DECAY = 1
TAU = 0.005
LR = 1e-5

n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)
# policy_net = DQN(n_actions=n_actions, input_height=160, input_width=210).to(device)
# target_net = DQN(n_actions=n_actions, input_height=160, input_width=210).to(device)
policy_net = DQN3(n_actions=n_actions, dropout=0).to(device)
target_net = DQN3(n_actions=n_actions, dropout=0).to(device)
# policy_net = Dueling_DQN_2016_Modified(n_actions=n_actions, dropout=0).to(device)
# policy_net = Dueling_DQN_2016_Modified(n_actions=n_actions, dropout=0).to(device)

# policy_net = Dueling_DQN_2016_Modified(num_classes=n_actions).to(device)
# target_net = Dueling_DQN_2016_Modified(num_classes=n_actions).to(device)

target_net.load_state_dict(policy_net.state_dict())
policy_net.eval()
target_net.eval()
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0
episodes_done = 0

def select_action(state):
    # if(state.shape[3] == 3):
    #     state = state.transpose(1, 3).transpose(2,3)
    # else:
    # breakpoint()
        # state = state_batch.transpose(1,3).transpose(2,3).shape
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    # if(steps_done > 250):
    #     eps_threshold = 0
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # Exploit
            # breakpoint()
            return policy_net(state).argmax(dim=1).to(device)  # exploit

            # return policy_net(state).max(1).indices.view(1, 1)
    else:
        # Explore
        return torch.tensor([env.action_space.sample()], device=device, dtype=torch.long)
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
    plt.ylabel("Score")
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
    
    # breakpoint()
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    
    # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state).unsqueeze(1)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.uint8)
    
    next_states = torch.cat([s for s in batch.next_state if s is not None])
    # breakpoint()
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(-1))
    
    
    # actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action))) 
    # rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward))) 
    # breakpoint()
    if(len(next_states.shape) == 3):
        next_states = next_states.unsqueeze(1)
    last_screens_of_state = next_states[:,-1,:,:] #(B,H,W)
    final_state_locations = last_screens_of_state.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
    non_final_state_locations = (final_state_locations == False)        
    non_final_states = next_states[non_final_state_locations] #(B',4,H,W)
    batch_size = next_states.shape[0]
    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_state_locations] = target_net(non_final_states).detach().max(dim=1)[0]
    # non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    
    # state_batch = torch.cat(batch.state).to(device)
    # action_batch = torch.cat(actions)
    # reward_batch = torch.cat(rewards)
    # state_action_values = policy_net(state_batch).gather(1, action_batch)
    
    
    # next_state_values = torch.zeros(BATCH_SIZE, device=device)
    # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    if(next_states.shape[0] != 128):
        # breakpoint()
        # next_states
        reward_batch = reward_batch[:next_states.shape[0]]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # with torch.no_grad():
    #     next_state_values[non_final_mask] = target_net(non_final_next_states)
    # breakpoint()
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values[:next_states.shape[0]], expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    # torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 5000
else:
    num_episodes = 50
current_reward = 0

def get_state():
    screen = env.render()
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    if len(screen.shape) == 1:
        screen = screen.view(1, -1)  # Reshape to 2D tensor
    bbox = [0,0,210,160] #(x,y,delta_x,delta_y)
    screen = screen[bbox[0]:bbox[2]+bbox[0], bbox[1]:bbox[3]+bbox[1]] #BZX:(CHW)
    resize = T.Compose([
        T.ToPILImage()
        , T.Grayscale()
        , T.Resize((210, 160)) #BZX: the original paper's settings:(110,84), however, for simplicty we...
        , T.ToTensor()
    ])
    # breakpoint()
    # screen = resize(screen).unsqueeze(0).to(device)
    return resize(screen).to(device)

for i_episode in range(num_episodes):
# while True:
    # if(keyboard.is_pressed('q')):
    #     break
    # Initialize the environment and get its state
    # state, info = env.reset()
    state = get_state()
    # breakpoint()
    # if(len(state.shape)):
    #     state = state.unsqueeze(0)
    total_reward = 0
    # state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        # breakpoint()
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        total_reward += reward
        done = terminated or truncated
        current_reward += reward.item()
        next_state = get_state()
        # Store the transition in memory
        # breakpoint()
        memory.push(state.clone(), action, next_state, torch.sign(reward))
        # memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        # for key in policy_net_state_dict:
        #     target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            scores.append(current_reward)
            episodes_done += 1
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
            print("eps_threshold", eps_threshold)
            print("Episode", episodes_done, "Reward", current_reward, "Steps", steps_done)
            current_reward = 0
            # policy_net.save("model.pth")
            plot_durations()
            break
torch.save(policy_net.state_dict(), "model.pth")
print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()