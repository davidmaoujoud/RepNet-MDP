import System
from Experiments import MDPParameters
import RepNetTree
import copy
import numpy as np
from Agent import Agent
import random


class RepNetAgent(Agent):
    """RepNet-MDP agent, extends the generic Agent class, this class implements a RepNet-MDP lookahead algorithm."""

    def __init__(self, g, system: System.System, parameters: MDPParameters.MDPParameters, epsilon=0.2):
        Agent.__init__(self)
        self.g = g
        self.system = system
        self.parameters = parameters
        self.number_of_agents = len(self.system.agents)
        self.number_of_actions_u = len(system.actions_u)
        self.rep_steps_per_transition = len(self.parameters.directed_transition_models[g][0][0][0])
        self.AD = [[[1 / len(system.actions) for a in range(len(system.actions))] for s in range(len(system.states))] for h in range(len(system.agents))]
        self.Img = [[0 for i in range(len(system.agents))] for h in range(len(system.agents))]
        for i in self.system.agents:
            self.Img[i][i] = 1
        self.tree: RepNetTree.Node = None

        self.epsilon = epsilon # Exploration-exploitation trade-off
        self.epsilon = 0
        self.eps = 0.1 # Used for ADE correction with deterministic transition models

        self.T2 = [[0,2,4,6],[1,3,5,7]]
        self.T1 = [[0,1,4,5],[2,3,6,7]]
        self.T0 = [[0,1,2,3],[4,5,6,7]]

    def T(self, h, s, a, sp, rep_h):
        """
        Transition function of agent g, combines directed and undirected transition functions
        h: sender of action a
        a: action sent by agent h
        s: current environment state
        sp: potential future environment state
        rep_h: reputation of agent h according to agent g
        """
        if a in self.system.actions_u:

            if self.number_of_agents == 2:
                probability_of_sp = 0
                if h != self.g:
                    for action in self.system.actions_u:
                        probability_of_sp += self.parameters.objective_transition_model[action*self.number_of_actions_u + a][s][0][sp]
                else:
                    for action in self.system.actions_u:
                        probability_of_sp += self.parameters.objective_transition_model[action + a*self.number_of_actions_u][s][0][sp]
                probability_of_sp /= self.number_of_actions_u
                return probability_of_sp
            elif self.number_of_agents == 3:
                probability_of_sp = 0
                if h == 0:
                    probability_of_sp = sum([self.parameters.objective_transition_model[i][s][0][sp] for i in self.T0[a]])
                elif h == 1:
                    probability_of_sp = sum([self.parameters.objective_transition_model[i][s][0][sp] for i in self.T1[a]])
                elif h == 2:
                    probability_of_sp = sum([self.parameters.objective_transition_model[i][s][0][sp] for i in self.T2[a]])
                probability_of_sp /= self.number_of_actions_u
                return probability_of_sp
        else:
            return self.parameters.directed_transition_models[h][s][a-self.number_of_actions_u][sp][self.position_in_DT(rep_h)]

    def position_in_DT(self, rep):
        d = 2/self.rep_steps_per_transition
        li = list(np.arange(-1 + d/2, 1, d))
        repp = min(li, key=lambda x:abs(x-rep))
        return li.index(repp)

    def ETI(self, h, i, s, AD, u=False):
        """
        Expected total impact that agent h has on agent i, with current environment state s and
        current action distribution function AD
        """
        eti = 0
        for a in self.system.actions:
            eti12 = self.parameters.delta_weight * AD[i][s][a] * self.system.I(h, i, s, a)  # impact on h
            eti21 = (1 - self.parameters.delta_weight) * AD[h][s][a] * self.system.I(i, h, s, a)   # impact on i
            eti += (eti12 + eti21)
        return eti

    def IE(self, s, Img, AD, u=False):
        """
        Image estimation function, updates image Img
        """
        Img_n = copy.deepcopy(Img)
        for h in self.system.agents:
            for i in self.system.agents:
                if h != i:
                    Img_n[h][i] = self.system.U(Img[h][i], self.ETI(i, h, s, AD))
        return Img_n

    def REP(self, h, Img):
        """
        Calculates the reputation of agent h according to agent g
        """
        n = [x for x in self.system.agents if x != h]
        if h == self.g: # self-reputation
            return (1/(self.number_of_agents-1)) * sum(Img[h][i] * Img[i][self.g] for i in n)
        else:
            return (1 / (self.number_of_agents)) * sum(Img[h][i] * Img[i][self.g] for i in self.system.agents)
        # return (1 / (self.number_of_agents - 1)) * sum(Img[h][i] for i in n)

    def ADE(self, s, sp, Img, AD, u=False):
        """
        Action distribution estimation function, updates action distribution AD
        """
        AD_n = copy.deepcopy(AD)
        for h in self.system.agents:
            # AD only needs to be updated for the current state since this is the only place we
            # get to learn new information
            rep_h = self.REP(h, Img)
            denom = sum((self.T(h, s, ap, sp, rep_h) * AD[h][s][ap] + self.eps) for ap in self.system.actions)
            for a in self.system.actions:
                # Avoid div 0
                if denom == 0:
                    AD_n[h][s][a] = self.eps
                else:
                    AD_n[h][s][a] = (self.T(h, s, a, sp, rep_h) * AD[h][s][a] + self.eps) / denom
        return AD_n

    def PI(self, s, a, AD):
        """
        Perceived immediate impact on agent g
        """
        number_of_agents = len(self.system.agents)
        agents = copy.deepcopy(self.system.agents)
        agents.remove(self.g)
        pi = self.system.I(self.g, self.g, s, a) + sum(sum(self.system.I(self.g, h, s, ap) * AD[h][s][ap] for ap in self.system.actions) for h in agents)
        return (1/number_of_agents) * pi

    def lookahead(self, s):
        """
        Performs a look-ahead search starting at state s, and returns the optimal action given said look-ahead search
        """
        if random.random() < self.epsilon:  # We explore
            # return random.choice(self.system.actions)
            return random.choice(self.system.actions_u)
        self.tree = self.tree = self.construct(s, self.AD, self.Img, self.parameters.lookahead_depth)
        return self.best_action()

    def construct(self, s, AD, Img, depth):
        if depth == 0:
            orNode = RepNetTree.ORNode(s, AD, Img)
            for a in self.system.actions:
                andNode = RepNetTree.ANDNode(a)
                andNode.value = self.PI(s, a, AD)
                orNode.children.append(andNode)
            orNode.value = max(self.PI(s, i, AD) for i in self.system.actions)
            return orNode
        else:
            tree = RepNetTree.ORNode(s, AD, Img)
            for restriction in self.parameters.restrictions[self.g]:
                if restriction[0] == s: # In this state you have to pick a directed action
                    andNode = RepNetTree.ANDNode(restriction[1])
                    andNode.value = self.PI(s, restriction[1], AD)
                    for sp in self.system.states:
                        AD_p = self.ADE(s, sp, Img, AD)
                        Img_p = self.IE(s, Img, AD)
                        child = self.construct(sp, AD_p, Img_p, depth-1)
                        andNode.value += self.parameters.decay * self.T(self.g, s, restriction[1], sp, self.REP(self.g, Img_p)) * child.value
                        andNode.children.append(child)
                    tree.children.append(andNode)
                    tree.value = tree.children[0].value
                    return tree

            for a in self.system.actions:
                andNode = RepNetTree.ANDNode(a)
                andNode.value = self.PI(s, a, AD)
                for sp in self.system.states:
                    AD_p = self.ADE(s, sp, Img, AD)
                    Img_p = self.IE(s, Img, AD)
                    child = self.construct(sp, AD_p, Img_p, depth-1)
                    andNode.value += self.parameters.decay * self.T(self.g, s, a, sp, self.REP(self.g, Img_p)) * child.value
                    andNode.children.append(child)
                tree.children.append(andNode)
            tree.value = max(tree.children[i].value for i in self.system.actions)
            return tree

    def best_action(self):
        """
        Returns the best action given the current look-ahead tree
        """
        # Directed transitions: When an agent is waiting for a response (e.g. after offering a trade), it needs to pick
        # the corresponding "wait" action
        for (state, action) in self.parameters.restrictions[self.g]:
            if self.system.current_state == state:
                return action

        # Find the best action, by picking the action associated with the highest q-value among the root node's children
        best_action = self.tree.children[0].a
        best_value = self.tree.children[0].value
        for andNode in self.tree.children:
            if andNode.value >= best_value:
                best_action = andNode.a
                best_value = andNode.value
        return best_action

    def update(self, sp):
        """
        Change the current environment state to sp
        """
        AD_p = self.ADE(self.system.current_state, sp, self.Img, self.AD, True)
        Img_p = self.IE(self.system.current_state, self.Img, self.AD, True)

        self.AD = AD_p
        self.Img = Img_p
