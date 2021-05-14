[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# Udacity-Continuous-Control-Project

The Jupyter-Notebook-File [Continuous_Control.ipynb](Continuous_Control.ipynb) includes my solution to the Continuous-Control-Project in the Deep-Reinforcement-Learning-Nanodegree.

### Introduction

In this project, an agent is trained in the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. In fact we train not just one agent but the training is distributed over 20 agents which train simultaneously. The training is done by adapting the DDPG algorithm  from the paper ["Continuous control with deep reinforcement learning"](https://arxiv.org/pdf/1509.02971.pdf) to this environment.

![Trained Agent][image1]

### Description of the Environment

In this environment, a double-jointed arm can move to target locations. A reward between 0 and +0.1 is provided for each step that the agent's hand is in or near the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment is said to be solved if the 20 agents achieve an average return of at least +30 over 100 consecutive episodes. Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30. 

### Installation

If you want to use the [Continuous_Control.ipynb](Continuous_Control.ipynb)-notebook to train agents on your own you first have to download several packages and the environment. 

To install all the necessary packages and dependencies you can set up a new python environment as explained in the README.md in the [DRLND Github repository](https://github.com/udacity/deep-reinforcement-learning#dependencies).
Depending on your operating system you can download the environment using one of the following links (the extraction of the file that I downloaded is the Reacher_Linux folder that is included in repository):

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
After downloading the file you have to unzip it. Then follow the instructions in the first part of the Jupyter Notebook to instantiate the environment. Note that in the code the visualization of the environment is turned off. Ofcourse this can be changed. 

### Instructions on [Continuous_Control.ipynb](Continuous_Control.ipynb)-notebook

After you instantiated the environment in the first part you will examine the state and action space in the second. In the third part you can check how random agents perform in the environment.

By running the cells in the fourth section you will train your own agents on the environment and test it afterwards.  
Once the necessary classes (Actor, Critic, Agent, OUNoise and ReplayBuffer) are defined you will set up an agent by creating an instance of the Agent-class. Then (by defining and running the ddpg()-function) the deep deterministic policy gradient (DDPG-) algorithm is carried out on this instance. 

The agents stop learning after the goal is reached (average total reward of +30 over 100 consecutive episodes). You can plot the performance of your agents that was achieved during training.

At the end of the notebook you can test your trained agents on 100 further episodes and check their performance by looking at the score-plot.



