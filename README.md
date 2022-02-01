[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"

# rl-playground
Trying different RL + DeepRL approaches on known benchmarks

## SoccerTwos: Actor-Critic Proximal Policy Optimization
Using the [SoccerTwos](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#soccer-twos) environment from the ML-Agents GitHub we use **Actor-Critic Proximal Policy Optimization** to train two agents to play soccer. 

![Soccer][image2]

To download the environment based on your operating system, Udacity DRLND provides the following links:
- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)

#### Requirements
This code runs successfuly on macOS Catalina (v.10.15.7). It is recommended to use a virtual environment with Python 3.6. Then all necessary requirements are installed by navigating to the [python](./python) folder, which is provided by the Udacity DRLND, and executing:
```
pip install .
```
within the virtual environment. 

## Fundamentals
This is a more tutorial-like section where classic techniques such as Policy Iteration, Value Iteration, Monte Carlo Control, and Policy Gradient are implemented and explored. For more details see the corresponding [folder](./fundamentals).