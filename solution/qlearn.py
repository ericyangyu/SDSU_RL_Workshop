import gym
import math
import random
import numpy as np


DISCOUNT = 0.99     # Discount Factor

# Actions
LEFT = 0
RIGHT = 1

BUCKETS = (10, 10,) # How many buckets we should have

class QLearningAlgorithm: 
	def __init__(self, env):
		# Environment
		self.env = env

		# Q table as {state: [value_1, ...]}
		self.q_table = {}

	# Discretize based on pole angle and pole angular velocity
	def discretize(self, s):
		# Define upper and lower bounds
		upper_bounds = [self.env.observation_space.high[2], math.radians(40)]
		lower_bounds = [self.env.observation_space.low[2], -math.radians(40)]

		bucket_s = []
		for i in range(len(s)):
			# Perform min-max normalization to squish all values between 0 and 1
			norm_s = (s[i] - lower_bounds[i]) / (upper_bounds[i] - lower_bounds[i])

			# Bucketize
			bucket_i = int(round((BUCKETS[i] - 1) * norm_s))
			bucket_i = min(BUCKETS[i] - 1, max(0, bucket_i))

			bucket_s.append(bucket_i)

		return tuple(bucket_s)

	# Annealing learning rate
	def alpha(self, num_eps):
		lr = 1 - math.log10((num_eps + 1)/25)

		if lr > 1:
			return 1
		elif lr < 0.1:
			return 0.1

		return lr

	# Annealing exploration epsilon
	def epsilon(self, num_eps):
		lr = 1 - math.log10((num_eps + 1)/25)

		if lr > 1:
			return 1
		elif lr < 0.1:
			return 0.1

		return lr

	def get_action(self, s, epsilon):
		if random.random() < epsilon:
			return self.env.action_space.sample()
			return random.sample([LEFT, RIGHT])
		else:
			return np.argmax(self.q_table[s])

	def learn(self, num_e):
		for episode in range(num_e):
			s = self.env.reset()
			s = self.discretize((s[2], s[3]))

			if self.q_table.get(s) is None:
				self.q_table[s] = [0,0]

			done = False
			time_steps = 0
			while not done:
				self.env.render()

				action = self.get_action(s, self.epsilon(episode))
				new_s, r, done, _ = self.env.step(action)
				new_s = self.discretize((new_s[2], new_s[3]))

				if self.q_table.get(new_s) is None:
					self.q_table[new_s] = [0,0]

				self.q_table[s][action] += (self.alpha(episode) * (r + DISCOUNT * max(self.q_table[new_s]) - self.q_table[s][action]))

				time_steps += 1

				if done:
					break

				s = new_s
			print(f"Episode {episode} done with {time_steps} time steps.")
		self.env.close()