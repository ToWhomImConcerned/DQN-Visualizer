# DQN Visualizer

![demo](DQN_Heatmap_amin.gif)

A deep reinforcement learning agent that learns to navigate a maze using 
Deep Q-Learning (DQN). Watch the neural network build its understanding of 
the environment in real time through a live heatmap that updates every step.

## How it works

Unlike traditional Q-learning which stores values in a lookup table, this agent 
uses a neural network to *approximate* Q-values. The network takes a three-layer 
state representation as input — agent position, goal position, and wall layout — 
and outputs a value for each possible action.

Training uses experience replay: the agent stores every step it takes in a replay 
buffer and trains on random batches of past experience. This breaks the correlation 
between consecutive steps and stabilizes learning.

The heatmap visualizes what the network has learned — warm colors mean the agent 
considers a cell valuable, cold colors mean the opposite. Watch the knowledge spread 
outward from the goal as training progresses.

## Features
- Neural network Q-value approximation with two hidden layers
- Experience replay buffer with 10,000 capacity
- Three-layer state encoding (agent, goal, walls)
- Manhattan distance reward shaping for denser feedback
- Live heatmap updated every training step
- Adjustable training speed with W/S keys

## Run it yourself
1. Clone the repo
2. Install dependencies: `pip install pygame numpy torch`
3. Run: `python visualizer.py`

## Built with
- Python
- PyTorch
- Pygame
- NumPy

## What's next
- Curriculum learning across randomly generated mazes
- Generalization to unseen maze layouts
- Multi-agent systems