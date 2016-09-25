import gym
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

env = gym.make('Breakout-v0')
env.reset()


def rgb_to_grey(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])

sample_period = 3
memory = 3
weight = [3,2,1]
out = np.zeros((210, 160))
j = 1
for i in range(3*sample_period):
    #env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    print(observation.shape)
    if i%sample_period == 0:
        print(out)
        print(rgb_to_grey(observation))
        out += j*rgb_to_grey(observation)
        j += 1
out / np.sum(weight)
plt.imshow(out, cmap='Greys_r')
plt.show()
    #print(np.dot(observation[...,:3], [0.299, 0.587, 0.114]))