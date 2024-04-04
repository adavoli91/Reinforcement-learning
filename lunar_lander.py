import numpy as np
from collections import deque
import gymnasium as gym
from torch import nn, tensor, optim, no_grad
from IPython import embed
import yaml
import pickle
import torch

config = yaml.load(open('./config.yaml'), Loader = yaml.FullLoader)
config_model = config['model']

class Model(nn.Module):
    def __init__(self, state_dim: int, action_num: int, config_model: dict) -> None:
        '''
        Class to define the model architecture.

        Args:
            state_dim: Dimension of the state space.
            action_num: Number of possible actions.
            config_model: Dictionary containing the relevant parameters for model definition.

        Returns: None
        '''
        super().__init__()
        list_n_neurons = config_model['n_neurons']
        # define hidden layers from the corresponding numbers of neurons
        list_hidden_layers = []
        for i in range(len(list_n_neurons)):
            if i == 0:
                layer = nn.Linear(in_features = state_dim, out_features = list_n_neurons[i])
            else:
                layer = nn.Linear(in_features = list_n_neurons[i-1], out_features = list_n_neurons[i])
            list_hidden_layers.append(layer)
        #
        self.layers_hidden = nn.ModuleList(list_hidden_layers)
        self.layer_relu = nn.ReLU()
        self.layer_out = nn.Linear(in_features = list_n_neurons[-1], out_features = action_num)

    def forward(self, x: tensor) -> tensor:
        '''
        Function to compute the action value function.

        Args:
            x: Batch of states observations.

        Returns:
            q: Action value functions for each of the states in the batch.
        '''
        for hidden in self.layers_hidden:
            x = hidden(x)
            x = self.layer_relu(x)
        q = self.layer_out(x)
        return q
    
class Memory:
    def __init__(self, config_model: dict) -> None:
        '''
        Class to build the memory buffer.

        Args:
            config_model: Dictionary containing the relevant parameters for model definition.

        Returns: None
        '''
        self.buffer = deque(maxlen = config_model['len_memory'])
        self.batch_size = config_model['batch_size']
        self.device = config_model['device']

    def sample(self) -> tuple[tensor, tensor, tensor, tensor, tensor]:
        '''
        Function to sample a batch of states observations.

        Args: None.

        Returns:
            batch_states_pre: Batch of old states.
            batch_actions: Batch of taken actions.
            batch_rewards: Batch of obtained rewards.
            batch_states_post: Batch of new states.
            batch_terminal_states: Batch of boolean values indicating whether the episode ended.
        '''
        # get batch indices
        idx_batch = np.random.choice(range(len(self.buffer)), self.batch_size, replace = False)
        # extract batches
        batch_states_pre = tensor(np.array([self.buffer[i][0] for i in idx_batch], dtype = np.float32)).to(self.device)
        batch_actions = tensor(np.array([self.buffer[i][1] for i in idx_batch], dtype = np.int64)).reshape(-1, 1).to(self.device)
        batch_rewards = tensor(np.array([self.buffer[i][2] for i in idx_batch], dtype = np.float32)).to(self.device)
        batch_states_post = tensor(np.array([self.buffer[i][3] for i in idx_batch], dtype = np.float32)).to(self.device)
        batch_terminal_states = tensor(np.array([self.buffer[i][4] for i in idx_batch], dtype = np.int64)).reshape(-1, 1).to(self.device)
        #
        return batch_states_pre, batch_actions, batch_rewards, batch_states_post, batch_terminal_states
    
    def append_obs(self, state_pre: tensor, action: tensor, reward: tensor, state_post: tensor, terminal_state: tensor) -> None:
        '''
        Function to append a set of observations to the memory buffer.

        Args:
            state_pre: Old state.
            action: Taken actions.
            reward: Obtained rewards.
            state_post: New states.
            terminal_state: Boolean value indicating whether the episode ended.

        Returns: None
        '''
        self.buffer.append([state_pre, action, reward, state_post, terminal_state])

class Agent:
    def __init__(self, environment: object, memory: deque, config_model: dict) -> None:
        '''
        Class to build the agent.

        Args:
            environment: Environment where the agent acts.
            memory: Memory buffer used for training.
            config_model: Dictionary containing the relevant parameters for model definition.

        Returns: None
        '''
        self.environment = environment
        self.memory = memory
        self.config_model = config_model
        self.batch_size = config_model['batch_size']
        self.device = config_model['device']
        state_dim = environment.observation_space.shape[0]
        self.action_num = environment.action_space.n
        # define models
        self.model_pred = Model(state_dim = state_dim, action_num = self.action_num, config_model = config_model).to(self.device)
        self.model_target = Model(state_dim = state_dim, action_num = self.action_num, config_model = config_model).to(self.device)
        # define optimizer
        self.optimzer = optim.Adam(self.model_pred.parameters(), lr = config_model['lr'])

    def act(self, state: tensor, epsilon: float) -> int:
        '''
        Function to choose an action to take.

        Args:
            state: Current state.
            epsilon: Parameter used for epsilon-greedy decision making.

        Returns:
            action: Chosen action.
        '''
        # perform epsilon-greedy actions
        if np.random.rand() > epsilon:
            # compute action value function, for given state
            self.model_pred.eval()
            with no_grad():
                q = self.model_pred(tensor(state.reshape(1, -1)).to(self.device))
            self.model_pred.train()
            # determine the corresponding action
            action = np.argmax(q.detach().numpy())
        else:
            # randomly choose an action
            action = np.random.choice(np.arange(self.action_num))
        #
        return action
    
    def step(self, state: tensor, action: int, timestep: int) -> tuple[tensor, int, tensor, float, bool]:
        '''
        Function to perform a step during training.

        Args:
            state: Current (old) state.
            action: Chosen action.
            timestep: Timestep within current episode.

        Returns:
            old_state: Old state.
            action: Chosen action.
            state: New state.
            reward: Obtained reward.
            terminal_state: Boolean value indicating whether the episode ended.
        '''
        n_timestep_update_target = config_model['n_timestep_update_target']
        #
        old_state = state
        # make a step within the environment
        state, reward, terminal_state, _, _ = self.environment.step(action)
        # update the memory buffer
        self.memory.append_obs(old_state, action, reward, state, terminal_state)
        # check if the buffer is long enough for training
        if len(self.memory.buffer) > self.batch_size:
            # model training
            if timestep%int(n_timestep_update_target/2) == 0:
                # get batch of observations
                batch_states_pre, batch_actions, batch_rewards, batch_states_post, batch_terminal_states = self.memory.sample()
                # train prediction model
                self.learn(batch_states_pre, batch_actions, batch_rewards, batch_states_post, batch_terminal_states)
            # target model update
            if timestep%n_timestep_update_target == 0:
                self.update_target()
        #
        return old_state, action, state, reward, terminal_state

    def learn(self, batch_states_pre: tensor, batch_actions: tensor, batch_rewards: tensor, batch_states_post: tensor,
              batch_terminal_states: tensor) -> None:
        '''
        Function to perform a step during training.

        Args:
            batch_states_pre: Batch of old states.
            batch_actions: Batch of taken actions.
            batch_rewards: Batch of obtained rewards.
            batch_states_post: Batch of new states.
            batch_terminal_states: Batch of boolean values indicating whether the episode ended.

        Returns: None.
        '''
        # predict action value function corresponding to old state
        pred = self.model_pred(batch_states_pre).gather(1, batch_actions)
        # predict action value function corresponding to new state
        target = self.model_target(batch_states_post).detach()
        # compute the target term (and set the action value function to 0 if the terminal state has been reached)
        target = target.max(axis = 1, keepdim = True).values*(1 - batch_terminal_states)
        target = batch_rewards.reshape(-1, 1) + self.config_model['gamma']*target
        # compute loss
        loss = nn.MSELoss()(pred, target)
        # perform training step
        self.optimzer.zero_grad()
        loss.backward()
        self.optimzer.step()

    def update_target(self) -> None:
        '''
        Function to update the weights of the target model.

        Args: None.

        Returns: None.
        '''
        self.model_target.load_state_dict(self.model_pred.state_dict())

def dqn(environment: object, config_model: dict) -> None:
    '''
    Function to implement deep Q-learning.

    Args:
        environment: Environment where the agent acts.
        config_model: Dictionary containing the relevant parameters for model definition.

    Returns: None.
    '''
    len_memory = config_model['len_memory']
    batch_size = config_model['batch_size']
    #
    memory = Memory(config_model = config_model)
    agent = Agent(environment = environment, memory = memory, config_model = config_model)
    # initialize epsilon
    epsilon_in = config_model['epsilon_in']
    epsilon_fin = config_model['epsilon_fin']
    n_ep_start_decr_eps = config_model['n_ep_start_decr_eps']
    n_ep_step_decr_eps = config_model['n_ep_step_decr_eps']
    n_ep_keep_eps_fixed = config_model['n_ep_keep_eps_fixed']
    # initialize lists for rewards storing
    list_rew_ep = []
    list_rew_ep_roll = deque(maxlen = 100)
    # loop over episodes
    for episode in range(config_model['n_episodes']):
        reward_ep = 0
        # compute epsilon for the given episode
        epsilon = epsilon_in
        if episode > n_ep_start_decr_eps:
            if episode < n_ep_keep_eps_fixed:
                round_episode = round(episode/n_ep_step_decr_eps)*n_ep_step_decr_eps
                time_factor_base = (epsilon_in - epsilon_fin)/epsilon_fin
                time_factor_exp = (n_ep_start_decr_eps - round_episode)/(n_ep_keep_eps_fixed - n_ep_start_decr_eps)
                epsilon = epsilon_fin + (epsilon_in - epsilon_fin)*time_factor_base**time_factor_exp
                epsilon_fixed = epsilon
            else:
                epsilon = epsilon_fixed
        # reset the environment
        state = environment.reset()
        if type(state) == tuple:
            state = state[0]
        # loop over timesteps within an episode
        for timestep in range(config_model['max_timestep_per_ep']):
            # select an action
            action = agent.act(state, epsilon)
            # perform a step
            old_state, action, state, reward, terminal_state = agent.step(state, action, timestep)
            # update the reward
            reward_ep += reward
            #
            if terminal_state == True:
                break
        list_rew_ep.append(reward_ep)
        list_rew_ep_roll.append(reward_ep)
        print(f'\rEpisode {episode}: epsilon = {epsilon}, episode reward = {np.mean(list_rew_ep_roll)}', end = '')
        if episode%100 == 0:
            torch.save(agent.model_pred.state_dict(), f'./artifacts/checkpoint_{episode}.pth')
            pickle.dump(list_rew_ep, open('./artifacts/list_rew_ep.pickle', 'wb'))
            print(f'\rEpisode {episode}: epsilon = {epsilon}, episode reward = {np.mean(list_rew_ep_roll)}')
        if np.mean(list_rew_ep_roll) >= 200:
            break
        epsilon = max(epsilon*0.995, 0.15)
    pickle.dump(list_rew_ep, open('./artifacts/list_rew_ep.pickle', 'wb'))
    torch.save(agent.model_pred.state_dict(), f'./artifacts/checkpoint_{episode}.pth')

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    dqn(environment = env, config_model = config_model)