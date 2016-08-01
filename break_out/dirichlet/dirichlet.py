from scipy.special import gamma, digamma
from scipy.stats import dirichlet
import numpy as np
import matplotlib.pyplot as plt
import pickle


alpha = [1,1,2]
mu = [0.05,0.05,0.95] # constraint must add to 1

def dirichlet(mu, alpha):
    mu = np.array(mu)
    alpha = np.array(alpha)
    product = np.product(mu ** (alpha - 1))
    normaliser = gamma(alpha.sum())/np.product(gamma(alpha))
    result = product * normaliser
    return result

print(dirichlet(mu, alpha))
print(dirichlet(mu, alpha))

print(np.random.dirichlet([1,1,100]))

#plt.plot()

x = np.linspace(0,4,100)
y = gamma(x+2) - gamma(x)
y = np.log(x)
#plt.plot(x, y)
#plt.draw()


def create_reward(a):
    e = np.random.rand()/100
    return -sum(np.abs(np.array([0.2-0.5*e,0.2-0.5*e,0.6+e]) - a))

theta = np.array([1.0,1.0,1.0])


def create_action(theta):
    action = np.random.dirichlet(theta)
    return action
#reward_score = create_reward(action)


def update_rule(action, reward, theta, step):
    update = np.array([0.0] * len(action))
    for i in range(len(action)):
        update[i] = step * (digamma(sum(theta)) - digamma(theta[i]) + np.log(action[i])) * reward
    return update

reward_list = []
for j in range(100000):
    print('iteration: {}'.format(j))
    action = create_action(theta)
    print('action: {}'.format(action))
    reward = create_reward(action)
    print('reward: {}'.format(reward))
    update = update_rule(action, reward, theta, 0.1)
    print('update: {}'.format(update))
    theta += update
    print('theta: {}'.format(theta))
    reward_list.append(reward)

pickle.dump(reward_list, open('results/rewards.p', 'wb'))

