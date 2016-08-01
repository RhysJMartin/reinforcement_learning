import gym
env = gym.make('MountainCar-v0')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)