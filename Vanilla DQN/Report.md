# Project Report

## Implementation

For this task, I have implemented a Vanilla DQN to train my agent to collect bananas. I obtained the code from the coding exercises and adapted it here to complete my project.

## Choice of Hyperparameters

The network architecture is consisted of: 2 hidden layers, each having 64 hidden units; 1 output layer with a single output for each valid action.

Following hyper parameters where used:
``` python
num_episode = 2000      # number of episodes ran
eps_start=1.0           # the starting value for epsilon
eps_end=0.01            # the final value for epsilon after decaying
eps_decay=0.995         # the epsilon decay rate
LR = 5e-4               # learning rate
BUFFER_SIZE = 100000    # replay memory size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
UPDATE_EVERY = 4        # how often should the network be updated.
```

## Plot of Rewards

Thought I trained the agent for 2000 episodes, the environment was solved at around 500 episodes. The average score peaks at around 900 episodes, reaching 16.58.  
<img src="Vanilla DQN/PlotOfRewards.png">



## Ideas for Future Work

For this project, I have only implemented the base model with the basic environment provided by Unity. Though it was able to solve the tasks, I would like to go far and beyond to experiment further with this project. 

- I plan to feed the network with pixel values as input rather than the 37 states provided. I will then add a few convolutional layers to my Vanilla DQN, train the agent and evaluate its performance.
- I plan to implement more advanced models such as the Dueling DQN and the Rainbow DQN and compare the performance between my Vanilla DQN with that of the more advanced models.
- I plan to make adjustments to the environment such as deploying more blue bananas while reducing the number of yellow bananas in the environment to make the task harder for my agent. I will then evaluate the performance of my agent at various difficulties using different models to compare its performance.


```python

```
