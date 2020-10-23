import System
from Experiments import MDPParameters
import copy
from Agent import Agent
import random


class QLearner(Agent):

    """Q-learning agent, extends the generic Agent class, this class implements an Epsilon-greedy
    Q-learning algorithm."""

    def __init__(self,
                 g,
                 system: System.System,
                 parameters: MDPParameters.MDPParameters,
                 epsilon=0.50,
                 learning_rate=0.1):
        Agent.__init__(self)
        self.g = g
        self.system = system
        self.parameters = parameters

        self.q_values = {}
        self.learning_rate = learning_rate
        self.decay = self.parameters.decay
        self.epsilon = epsilon
        self.q_val_init = 1

        self.actions = self.parameters.actions_u
        self.s = 0
        self.a = 0

        self.AD = [[[1 / len(system.actions) for a in range(len(system.actions))] for s in range(len(system.states))] for h in range(len(system.agents))]

    def R(self, s, a):
        number_of_agents = len(self.system.agents)
        agents = copy.deepcopy(self.system.agents)
        agents.remove(self.g)
        pi = self.system.I(self.g, self.g, s, a) + sum(sum(self.system.I(self.g, h, s, ap) * self.AD[h][s][ap] for ap in self.system.actions) for h in agents)
        return (1/number_of_agents) * pi

    def update(self, new_state):
        reward = self.R(self.s, self.a) # Reward after taking step, so using previous state and current action

        old_q_value = self.q_values.get((self.s, self.a), self.q_val_init) # try to get, else init
        q_max = max([self.q_values.get((new_state, _action), self.q_val_init) for _action in self.actions])
        new_q_value = old_q_value + self.learning_rate * (reward + self.decay * q_max - old_q_value)

        # Update the Q value
        self.q_values[self.s, self.a] = new_q_value
        self.s = new_state

    def lookahead(self, state):
        if random.random() < self.epsilon:  # We explore
            #action = random.choice(self.actions)
            action = random.choice([0,1])
        else: # Exploitation
            action = max(self.actions, key=lambda action: self.q_values.get((state, action), self.q_val_init))
        self.a = action
        return action
