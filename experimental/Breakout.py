import time
import gymnasium as gym
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim 
import torchvision.transforms as T
import optuna

# hyperparameters
# num_target_update = 0 # base: 0
# current_step = 0 # base: 0
# startpoint = 50000 # base: 50000
# endpoint = 1000000 # base: 1000000
# kneepoint = 1000000 # base: 1000000
# start = 1	 # base: 1
# end = .1 # base: 0.1
# final_eps = 0.01 # base: 0.01
# final_knee_point = 22000000 # base: 22000000
# action_repeat = 4 # base: 4
# batch_size = 32 # base: 32
# replay_start_size = 50000 # base: 50000
# gamma = 1 # base: 1
# max_iteration = 500000 # base: 500000
# max_frames = 22000000 # base: 22000000
# target_update = 2500 # base:  2500
# learning_rate=0.0000625 # base: 0.0000625

num_target_update = 0 # base: 0
current_step = 0 # base: 0
startpoint = 2 # base: 50000
endpoint = 1000000 # base: 1000000
kneepoint = 1000000 # base: 1000000
start = 0.14712368077870006	 # base: 1
end = .1 # base: 0.1
final_eps = 0.01 # base: 0.01
final_knee_point = 22000000 # base: 22000000
action_repeat = 4 # base: 4
batch_size = 84 # base: 32
replay_start_size = 50000 # base: 50000
gamma = 0.9529810958690669 # base: 1
max_iteration = 500000 # base: 500000
max_frames = 22000000 # base: 22000000
target_update = 2500 # base:  2500
learning_rate=0.0000665660642462102 # base: 0.0000625

Eco_Experience = namedtuple(
    'Eco_Experience',
    ('state', 'action', 'reward')
)
Experience = namedtuple(
    'Experience',
    ('state', 'action', 'next_state', 'reward')
)
class ReplayMemory():
    # save one state per experience to improve memory size
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.push_count = 0
        self.dtype = torch.uint8

    def push(self, experience):
        state = (experience.state * 255).type(self.dtype).cpu()
        new_experience = Eco_Experience(state,experience.action,experience.reward)

        if len(self.memory) < self.capacity:
            self.memory.append(new_experience)
        else:
            self.memory[self.push_count % self.capacity] = new_experience
        self.push_count += 1

    def sample(self, batch_size):
        experience_index = np.random.randint(3, len(self.memory)-1, size = batch_size)
        experiences = []
        for index in experience_index:
            if self.push_count > self.capacity:
                state = torch.stack(([self.memory[index+j].state for j in range(-3,1)])).unsqueeze(0)
                next_state = torch.stack(([self.memory[index+1+j].state for j in range(-3,1)])).unsqueeze(0)
            else:
                state = torch.stack(([self.memory[np.max(index+j, 0)].state for j in range(-3,1)])).unsqueeze(0)
                next_state = torch.stack(([self.memory[np.max(index+1+j, 0)].state for j in range(-3,1)])).unsqueeze(0)
            experiences.append(Experience(state.float().cuda()/255, self.memory[index].action, next_state.float().cuda()/255, self.memory[index].reward))
        return experiences

    def can_provide_sample(self, batch_size, replay_start_size):
        return (len(self.memory) >= replay_start_size) and (len(self.memory) >= batch_size + 3)


class DQN(nn.Module):
    def __init__(self, num_classes=4, init_weights=True):
        super().__init__()

        self.cnn = nn.Sequential(nn.Conv2d(4, 32, kernel_size=8, stride=4),
                                        nn.ReLU(True),
                                        nn.Conv2d(32, 64, kernel_size=4, stride=2),
                                        nn.ReLU(True),
                                        nn.Conv2d(64, 64, kernel_size=3, stride=1),
                                 nn.ReLU(True)
                                        )
        self.classifier = nn.Sequential(nn.Linear(7*7*64, 512),
                                        nn.ReLU(True),
                                        nn.Linear(512, num_classes)
                                        )
        if init_weights:
            self._initialize_weights()
    def forward(self, x):
        x = self.cnn(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
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

tracker_dict = {}
tracker_dict["minibatch_updates_counter"] = 1
tracker_dict["actions_counter"] = 1
tracker_dict["running_reward"] = 0
tracker_dict["rewards_hist"] = []
tracker_dict["loss_hist"] = []
tracker_dict["eval_model_list_txt"] = []
tracker_dict["rewards_hist_update_axis"] = []
# only used in evaluation script
tracker_dict["eval_reward_list"] = []
tracker_dict["best_frame_for_gif"] = []
tracker_dict["best_reward"] = 0
class AtariEnvManager():
    def __init__(self, device, game_env, is_use_additional_ending_criterion):
        self.device = device
        self.game_env = game_env
        self.env = gym.make(game_env, render_mode="rgb_array").unwrapped
        self.env = gym.wrappers.RecordVideo(env=self.env, video_folder="videos", name_prefix="Breakout", video_length=0, episode_trigger=lambda x: x % 100 == 0)
        self.env.reset()
        self.current_screen = None
        self.done = False
        # running_K: stacked K images together to present a state
        # running_queue: maintain the latest running_K images
        self.running_K = 4
        self.running_queue = []
        self.is_additional_ending = False # This may change to True along the game. 2 possible reason: loss of lives; negative reward.
        self.current_lives = None
        self.is_use_additional_ending_criterion = is_use_additional_ending_criterion

    def reset(self):
        self.env.reset()
        self.current_screen = None
        self.running_queue = [] # clear the state
        self.is_additional_ending = False

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render()

    def num_actions_available(self):
        return self.env.action_space.n

    def print_action_meanings(self):
        print(self.env.get_action_meanings())

    def take_action(self, action):
        _, reward, self.done, _, lives = self.env.step(action.item())
        if self.is_use_additional_ending_criterion:
            self.is_additonal_ending_criterion_met(lives,reward)
        return torch.tensor([reward], device=self.device)

    def just_starting(self):
        return self.current_screen is None
    def is_initial_action(self):
        return sum(self.running_queue).sum() == 0

    def init_running_queue(self):
        """
        initialize running queue with K black images
        :return:
        """
        self.current_screen = self.get_processed_screen()
        black_screen = torch.zeros_like(self.current_screen)
        for _ in range(self.running_K):
            self.running_queue.append(black_screen)

    def get_state(self):
        if self.just_starting():
            self.init_running_queue()
        elif self.done or self.is_additional_ending:
            self.current_screen = self.get_processed_screen()
            black_screen = torch.zeros_like(self.current_screen)
            # BZX: update running_queue
            self.running_queue.pop(0)
            self.running_queue.append(black_screen)
        else: #BZX: normal case
            s2 = self.get_processed_screen()
            self.current_screen = s2
            # BZx: update running_queue with s2
            self.running_queue.pop(0)
            self.running_queue.append(s2)

        return torch.stack(self.running_queue,dim=1).squeeze(2) #BZX: check if shape is (1KHW)

    def is_additonal_ending_criterion_met(self,lives,reward):
        "for different atari game, design different ending state criterion"
        if self.game_env == "BreakoutDeterministic-v4":
            if self.is_initial_action():
                self.current_lives = lives['lives']
            elif lives['lives'] < self.current_lives:
                self.is_additional_ending = True
            else:
                self.is_additional_ending = False
            self.current_lives = lives['lives']
        if self.game_env == "PongDeterministic-v4":
            if reward < 0: #miss one ball will lead to a ending sate.
                self.is_additional_ending = True
            else:
                self.is_additional_ending = False
        return False

    def get_screen_height(self):
        screen = self.get_processed_screen()
        return screen.shape[2]

    def get_screen_width(self):
        screen = self.get_processed_screen()
        return screen.shape[3]

    def get_processed_screen(self):
        # plt.show()
        screen = self.render('rgb_array').transpose((2, 0, 1))  # PyTorch expects CHW
        
        # screen = self.render()  # PyTorch expects CHW
        # screen = self.crop_screen(screen)
        return self.transform_screen_data(screen) #shape is [1,1,110,84]

    def crop_screen(self, screen):
        # Ensure screen is a 2D tensor
        if len(screen.shape) == 1:
            screen = screen.view(1, -1)  # Reshape to 2D tensor

        if self.game_env == "BreakoutDeterministic-v4" or self.game_env == "PongDeterministic-v4":
            bbox = [34,0,160,160] #(x,y,delta_x,delta_y)
            screen = screen[bbox[0]:bbox[2]+bbox[0], bbox[1]:bbox[3]+bbox[1]] #BZX:(CHW)
        if self.game_env == "GopherDeterministic-v4":
            bbox = [110,0,120,160]
            screen = screen[bbox[0]:bbox[2]+bbox[0], bbox[1]:bbox[3]+bbox[1]]
        return screen

    def transform_screen_data(self, screen):
        # Convert to float, rescale, convert to tensor
        # breakpoint()
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # screen = self.crop_screen(screen)
        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            , T.Grayscale()
            , T.Resize((84, 84)) #BZX: the original paper's settings:(110,84), however, for simplicty we...
            , T.ToTensor()
        ])
        # add a batch dimension (BCHW)
        screen = resize(screen)

        return screen.unsqueeze(0).to(self.device)   # BZX: Pay attention to the shape here. should be [1,1,84,84]

def get_current(policy_net, states, actions):
    return policy_net(states).gather(dim=1, index=actions.unsqueeze(-1))

    
def DQN_get_next(target_net, next_states, mode = "stacked"):
    if mode == "stacked":
        last_screens_of_state = next_states[:,-1,:,:] #(B,H,W)
        final_state_locations = last_screens_of_state.flatten(start_dim=1).max(dim=1)[0].eq(0).type(torch.bool)
        non_final_state_locations = (final_state_locations == False)
        non_final_states = next_states[non_final_state_locations] #(B',4,H,W)
        batch_size = next_states.shape[0]
        # print("# of none terminal states = ", batch_size)
        values = torch.zeros(batch_size).to(device)
        if non_final_states.shape[0]==0: # BZX: check if there is survival
            print("EXCEPTION: this batch is all the last states of the episodes!")
            return values
        with torch.no_grad():
            values[non_final_state_locations] = target_net(non_final_states).detach().max(dim=1)[0]
        return values

def get_moving_average(period, values):
    # breakpoint()
    values = values.clone().detach().cpu()
    # values = torch.tensor(values, dtype=torch.float)
    if len(values) >= period:
        moving_avg = values.unfold(dimension=0, size=period, step=1).mean(dim=1).flatten(start_dim=0)
        moving_avg = torch.cat((torch.zeros(period-1), moving_avg))
        return moving_avg.numpy()
    else:
        moving_avg = torch.zeros(len(values))
        return moving_avg.numpy()
moving_avg = []
def plot(values, moving_avg_period):
    """
    test: plot(np.random.rand(300), 100)
    :param values: numpy 1D vector
    :param moving_avg_period:
    :return: None
    """
    global moving_avg
        # breakpoint()
    # Convert values to tensor if it's not already a tensor
    if not isinstance(values, torch.Tensor):
        values = torch.tensor(values)
        
    # Move tensor to CPU if it's on GPU
    if values.device.type == 'cuda':
        values = values.cpu()

    # plt.figure()
    plt.clf()
    plt.ylim(0, 100)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(values)
    moving_avg = get_moving_average(moving_avg_period, values)
    plt.plot(moving_avg)
    # print("Episode", len(values), "\n",moving_avg_period, "episode moving avg:", moving_avg[-1])
    plt.pause(0.0005)
    return moving_avg[-1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = AtariEnvManager(device, "BreakoutDeterministic-v4", is_use_additional_ending_criterion=True)

memory = ReplayMemory(1000000)
policy_net = DQN(num_classes = env.env.action_space.n).to(device)
target_net = DQN(num_classes = env.env.action_space.n).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
criterion = torch.nn.SmoothL1Loss()

 
plt.figure()
global t1, t2
t1, t2 = time.time(), time.time()


def get_exploration_rate(current_step):
    if(current_step < startpoint):
        return 1
    mid_seg = end + np.maximum(0, (1-end)-(1-end)/kneepoint * (current_step - startpoint))
    if(not final_eps):
        return mid_seg
    else:
        if(final_eps and final_knee_point and (current_step < kneepoint)):
            return mid_seg
        else:
            return final_eps + \
                       (end - final_eps)/(final_knee_point - kneepoint)*(final_knee_point - current_step)

def select_action(state, policy_net):
    global current_step
    rate = get_exploration_rate(current_step)
    current_step += 1
    if(rate > random.random()):
        action = random.randrange(env.env.action_space.n)
        return torch.tensor([action]).to(device)
    else:
        with torch.no_grad():
            return policy_net(state).argmax(dim=1).to(device)

# surface = pygame.display.set_mode((160, 210), pygame.DOUBLEBUF)
frames = []
env.env.start_video_recorder()
def single_episode():
    global num_target_update, current_step, startpoint, endpoint, kneepoint, start, end, final_eps, final_knee_point, action_repeat, batch_size, replay_start_size, gamma, max_iteration, max_frames, target_update, learning_rate, t1, t2
    env.reset()
    state = env.get_state()
    tol_reward = 0
    while(1):
        
        action = select_action(state, policy_net)
        reward = env.take_action(action)
        tol_reward += reward
        tracker_dict["actions_counter"] += 1
        
        next_state = env.get_state()
        memory.push(Experience(state[0, -1, :, :].clone(), action, "", torch.sign(reward)))  
        state = next_state
        
        if(current_step % action_repeat == 0) and \
            memory.can_provide_sample(batch_size, replay_start_size):
            experiences = memory.sample(batch_size)
            batch = Experience(*zip(*experiences))
            states = torch.cat(batch.state) 
            actions = torch.cat(batch.action)
            rewards = torch.cat(batch.reward)
            next_states = torch.cat(batch.next_state)
            if(len(states.shape) != 4 or len(next_states.shape) != 4):
                print("state shape = ", states.shape)
                print("next_state shape = ", next_states.shape)
            current_q_values = get_current(policy_net, states, actions)
            next_q_values = DQN_get_next(target_net, next_states)
            target_q_values = (next_q_values * gamma) + rewards
            optimizer.zero_grad()
            loss = criterion(current_q_values, target_q_values.unsqueeze(1))
            loss.backward()
            optimizer.step()
            tracker_dict["loss_hist"].append(loss.item())
            tracker_dict["minibatch_updates_counter"] += 1
            
            if(tracker_dict["minibatch_updates_counter"] % target_update == 0):
                target_net.load_state_dict(policy_net.state_dict())
                num_target_update += 1
                if num_target_update % 2 == 0 : t1 = time.time()
                if num_target_update % 2 == 1 : t2 = time.time()
                print("=" * 50)
                remaining_update_times = (max_iteration - tracker_dict["minibatch_updates_counter"])// \
                                  target_update
                time_sec = abs(t1-t2) * remaining_update_times
                print("estimated remaining time = {}h-{}min".format(time_sec//3600,(time_sec%3600)//60))
                print("len of replay memory:", len(memory.memory))
                print("minibatch_updates_counter = ", tracker_dict["minibatch_updates_counter"])
                print("current_step of agent = ", current_step)
                print("exploration rate = ", get_exploration_rate(current_step))
                print("=" * 50)
            if(tracker_dict["minibatch_updates_counter"] % 20000 == 0):
                torch.save(policy_net.state_dict(), "policy_net.pth")
                tracker_dict["eval_model_list_txt"].append("policy_net.pth")
            
        if env.done:
            
            tracker_dict["rewards_hist"].append(tol_reward)
            tracker_dict["rewards_hist_update_axis"].append(tracker_dict["actions_counter"])
            tracker_dict["running_reward"] = plot(tracker_dict["rewards_hist"], 100)
            return tol_reward.item()
            # break
            

trialCount = 0
def all_episodes(trial=None):
    global batch_size, startpoint, start, learning_rate, gamma, trialCount, policy_net, target_net, memory, optimizer, criterion
    
    memory = ReplayMemory(1000000)
    policy_net = DQN(num_classes = env.env.action_space.n).to(device)
    target_net = DQN(num_classes = env.env.action_space.n).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    criterion = torch.nn.SmoothL1Loss()
    # tracker_dict.cpu()
    if(trial is not None):
        batch_size = trial.suggest_int('batch_size', 32, 128)
        startpoint = trial.suggest_int('startpoint', 0, 100)
        start = trial.suggest_float('start', 0, 1)
        learning_rate = trial.suggest_float('learning_rate', 0.0000625, 0.0001)
        gamma = trial.suggest_float('gamma', 0.75, 1)
    tol_reward = None
    for episode in range(5000):
        # print(episode)
        if(tol_reward is None):
            tol_reward = single_episode()
        else:
            tol_reward = tol_reward *.999 + single_episode() * 0.001
        print("episode = ", episode, "reward = ", tol_reward)
    
    plt.figure()
    plt.ylim(0, 350)
    plt.plot(torch.tensor(tracker_dict["rewards_hist"]).cpu())
    plt.plot(moving_avg)
    plt.title("reward")
    plt.xlabel("episodes")
    plt.savefig("optuna val " + str(trialCount) + ".jpg")
    print("final moving average: " + str(moving_avg[-1]))
    trialCount += 1
    tracker_dict["minibatch_updates_counter"] = 1
    tracker_dict["actions_counter"] = 1
    tracker_dict["running_reward"] = 0
    tracker_dict["rewards_hist"] = []
    tracker_dict["loss_hist"] = []
    tracker_dict["eval_model_list_txt"] = []
    tracker_dict["rewards_hist_update_axis"] = []
    # only used in evaluation script
    tracker_dict["eval_reward_list"] = []
    tracker_dict["best_frame_for_gif"] = []
    tracker_dict["best_reward"] = 0
    return tol_reward


tune = False
if tune:
    study = optuna.create_study(storage="sqlite:///db1.sqlite3", direction='maximize', study_name="hyperparams0", load_if_exists=True)
    study.optimize(all_episodes, n_trials=100)
    print(study.best_params)
else:
    all_episodes()
env.env.close_video_recorder()
env.close()

with open("tracker_dict.txt", "w") as file:
    file.write(str(tracker_dict["eval_model_list_txt"]))
