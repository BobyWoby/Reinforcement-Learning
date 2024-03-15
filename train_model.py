import torch
from utilities import DQN, ReplayMemory, Transition, optimize_model, plot_durations
import gymnasium as gym
import torch.optim as optim
import matplotlib.pyplot as plt
import random
import math
from itertools import count


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('ALE/Breakout-v5', render_mode='human', obs_type='grayscale', )
env.metadata['render_fps'] = 60

env.reset()
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

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
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
plot_durations(show_result=True, scores=scores)
plt.ioff()
plt.show()