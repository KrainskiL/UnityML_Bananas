# Report for Banana Navigation

## State and action space, rewards in environment
The simulation contains a single agent navigating in a large, square environment.

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has **37 dimensions** and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. **Four discrete actions** are available:

- `0` - walk forward 
- `1` - walk backward
- `2` - turn left
- `3` - turn right

The task is episodic, and in order to solve the environment, your agent must get an average score of **+13** over 100 consecutive episodes.

## Learning algorithm

The agent training is conducted using `dqn` function in the `Navigation.ipynb` notebook. 

Agent is trained episodically until `n_episodes` is reached or until the environment is solved. Each episode continues until `max_t` time-steps is reached or until the environment is done.

Agent class is contained in `dqn_agent.py` file and utilize neural network defined in `model.py` file.

Additionaly Agent used following mechanisms:
* Epsilon-greedy action selection
* Replay buffer of experiences
* Soft targets with specified "softness" and update period
* Discounted rewards

DQN Agent Hyper Parameters:

- BUFFER_SIZE (int): replay buffer size
- BATCH_SIZE (int): mini batch size
- GAMMA (float): discount factor
- TAU (float): for soft update of target parameters
- LR (float): learning rate for optimizer
- UPDATE_EVERY (int): how often to update the network

Where 
`BUFFER_SIZE = int(1e5)`, `BATCH_SIZE = 128`, `GAMMA = 0.99`, `TAU = 1e-3`, `LR = 0.0001` and `UPDATE_EVERY = 3`  

Used hyperparameters values were found by trial and error until satisfactory results were reached.

### DQN Hyper Parameters  

- n_episodes (int): maximum number of training episodes
- max_t (int): maximum number of timesteps per episode
- eps_start (float): starting value of epsilon, for epsilon-greedy action selection
- eps_end (float): minimum value of epsilon
- eps_decay (float): multiplicative factor (per episode) for decreasing epsilon

Where
`n_episodes=1000`, `max_t=3000`, `eps_start=0.9`, `eps_end=0.01` and `eps_decay=0.99`

Used hyperparameters values were found by trial and error until satisfactory results were reached.

### Neural Network
The NN model utilize 2 x 64 Fully Connected Layers with Relu activation followed by a final Fully Connected layer with the same number of units as the action size. The network has an initial dimension the same as the state size. 

![NN Architecture](https://github.com/KrainskiL/UnityML_Bananas/blob/master/img/NN_arch.png?raw=true)

Used architecture is the same as in [`solution/Deep_Q_Network_Solution.ipynb`](https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/Deep_Q_Network_Solution.ipynb) 

## Plot of score

![Score](https://github.com/KrainskiL/UnityML_Bananas/blob/master/img/Scores.PNG?raw=true)

```
Episode 100	Average Score: 3.12
Episode 200	Average Score: 8.55
Episode 300	Average Score: 12.02
Episode 344	Average Score: 13.05
Environment solved in 244 episodes!	Average Score: 13.05
```

## Ideas for improvements

Hyperparameters for both DQN Agent and DQN training may be explored more systematically e.g. grid search but that requires much more computing power. GPU usage is suggested. 

As for algorithm itself, it can be enhanced using some of the ["Rainbow upgrades"](https://arxiv.org/abs/1710.02298) e.g. [dueling DQN](https://arxiv.org/abs/1511.06581) or [Noisy DQN](https://arxiv.org/abs/1706.10295)



