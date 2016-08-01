import pickle
import matplotlib.pyplot as plt
import pandas as pd

rewards = pickle.load(open('results/rewards.p', 'rb'))
# plt.plot(rewards)
# plt.show()
# print(len(rewards))
rewards = pd.Series(rewards)

plt.plot(pd.rolling_mean(rewards, 1000))
plt.show()

