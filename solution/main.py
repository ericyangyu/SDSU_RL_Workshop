import gym

from qlearn import QLearningAlgorithm


env = gym.make('CartPole-v1')

model = QLearningAlgorithm(env)
model.learn(10000000)