import random
import matplotlib.pyplot as plt
import csv


class OnlineSolver:
    """Coordinates the execution of the selected experiment and tracks variables of interest
    for the selected experiment before storing the results in CSV files."""

    def __init__(self, system, agents, parameters, experiment_number=1):
        self.system = system
        self.agents = agents
        self.parameters = parameters
        self.state_history = [self.system.current_state]
        self.best_actions = []
        self.tracked_variables = []
        self.tracked_variables2 = []
        self.test = ""
        self.experiment_number = experiment_number

    def execution(self, best_actions, o=0):
        # Directed actions need to be switched back to undirected actions
        for t, a in enumerate(self.system.agents):
            for (i, j) in self.parameters.directed_undirected_equivalence[t]:
                for k, a in enumerate(best_actions):
                    if a == i:
                        best_actions[k] = j

        print("Execution...")
        # Applying each agent's action to the environment and yielding the new environment state
        biased_state_list = []
        accumulator = 0
        for sp in self.system.states:
            number_of_actions = len(self.system.actions_u)
            number_of_agents = len(self.system.agents)
            probability_of_sp = 0

            if number_of_agents == 2:
                probability_of_sp = self.parameters.objective_transition_model[best_actions[0]*number_of_actions + best_actions[1]][self.system.current_state][0][sp]
            elif number_of_agents == 3:
                probability_of_sp = self.parameters.objective_transition_model[best_actions[0]*number_of_actions*2 + best_actions[1]*number_of_actions + best_actions[2]][self.system.current_state][0][sp]

            biased_state_list.append(probability_of_sp + accumulator)
            accumulator += probability_of_sp
        biased_random = random.uniform(0, biased_state_list[len(biased_state_list)-1])
        next_state = -1
        i = 0
        while next_state == -1:
            if biased_random <= biased_state_list[i]:
                next_state = i
            i += 1

        if self.test == "ABCTrade" and o >= 66 and self.system.current_state == 0:
            # Forces interaction between agents B and C (so that A has to learn indirectly)
            if random.random() < 0.50:
                next_state = 5
            else:
                next_state = 8
        elif self.test == "ABCTrade" and self.system.current_state == 0:
            # No trades are done by agents B and C
            if best_actions[0] == 0:
                next_state = 1 # Trade with B
            else:
                next_state = 2 # Trade with C

        self.state_history.append(next_state)

        # Send the new environment state to each agent
        for g in self.agents:
            g.update(next_state)

        # Update the system with the new environment state
        self.system.current_state = next_state

    def planning(self):
        print("Planning...")
        best_actions = []
        for g in self.agents:
            best_actions.append(g.lookahead(self.system.current_state))
        self.best_actions = best_actions
        return best_actions

    def online_repnet_solver(self):
        self.initialize_tracking()
        # Alternating between planning and execution
        for k in range(self.parameters.steps):
            best_actions = self.planning()
            self.track(k)
            self.execution(best_actions, k)
        plt.plot(self.state_history)
        plt.show()
        self.unpack()

    def initialize_tracking(self):
        if self.experiment_number == 1:
            self.test = "ABTrade"
        else:
            self.test = "ABCTrade"
            self.tracked_variables2 = [0 for i in range(self.parameters.steps)]

    def track(self, k=0):
        """Tracks the variables of interest for the selected experiment."""
        if self.experiment_number == 1:
            # Trade between agents A and B
            proba_B_accept_in_2 = self.agents[0].AD[1][2][3]
            proba_B_accept_in_3 = self.agents[0].AD[1][3][3]
            proba_B_refuse_in_2 = self.agents[0].AD[1][2][4]
            proba_B_refuse_in_3 = self.agents[0].AD[1][3][4]
            img_A_has_of_B = self.agents[0].Img[1][0]
            img_B_has_of_A = self.agents[0].Img[0][1]
            rep_A = self.agents[0].REP(0, self.agents[0].Img)
            rep_B = self.agents[0].REP(1, self.agents[0].Img)

            trade_offers = (self.system.current_state, self.best_actions[0])

            trade_offer = 0
            if self.system.current_state == 0 or self.system.current_state == 1:
                if self.best_actions[0] == 1:
                    trade_offer = 1

            self.tracked_variables.append((proba_B_accept_in_2, proba_B_accept_in_3, proba_B_refuse_in_2,
                                           proba_B_refuse_in_3, img_A_has_of_B, img_B_has_of_A, rep_A, rep_B,
                                           trade_offers, trade_offer))

        else:
            # Trade between agents A, B, and C
            proba_B_accept = self.agents[0].AD[1][1][0]
            proba_C_accept = self.agents[0].AD[2][2][0]

            rep_B = self.agents[0].REP(1, self.agents[0].Img)
            rep_C = self.agents[0].REP(2, self.agents[0].Img)

            if self.system.current_state == 0:
                action = self.best_actions[0]
                self.tracked_variables2[k] = action
            else:
                action = self.tracked_variables2[k - 1]
                self.tracked_variables2[k] = action

            self.tracked_variables.append((proba_B_accept, proba_C_accept, rep_B, rep_C, action))

    def unpack(self):
        """Save tracked variables in CSV files"""
        if self.experiment_number == 1:
            # Trade between agents A and B
            proba_B_accept_in_2 = [tuple[0] for tuple in self.tracked_variables]
            proba_B_accept_in_3 = [tuple[1] for tuple in self.tracked_variables]

            proba_B_refuse_in_2 = [tuple[2] for tuple in self.tracked_variables]
            proba_B_refuse_in_3 = [tuple[3] for tuple in self.tracked_variables]

            img_A_has_of_B = [tuple[4] for tuple in self.tracked_variables]
            img_B_has_of_A = [tuple[5] for tuple in self.tracked_variables]

            rep_A = [tuple[6] for tuple in self.tracked_variables]
            rep_B = [tuple[7] for tuple in self.tracked_variables]

            trade_offer = [tuple[9] for tuple in self.tracked_variables]

            self.write_to_csv("tradeABaccept2", proba_B_accept_in_2)
            self.write_to_csv("tradeABaccept3", proba_B_accept_in_3)
            self.write_to_csv("tradeABrefuse2", proba_B_refuse_in_2)
            self.write_to_csv("tradeABrefuse3", proba_B_refuse_in_3)
            self.write_to_csv("tradeABimgAB", img_A_has_of_B)
            self.write_to_csv("tradeABimgBA", img_B_has_of_A)
            self.write_to_csv("tradeABrepA", rep_A)
            self.write_to_csv("tradeABrepB", rep_B)
            self.write_to_csv("tradeABfreq", trade_offer)

        else:
            # Trade between agents A, B, and C
            proba_B_accept = [tuple[0] for tuple in self.tracked_variables]
            proba_C_accept = [tuple[1] for tuple in self.tracked_variables]

            rep_B = [tuple[2] for tuple in self.tracked_variables]
            rep_C = [tuple[3] for tuple in self.tracked_variables]
            actions = [tuple[4] for tuple in self.tracked_variables]

            self.write_to_csv("tradeABCacceptB", proba_B_accept)
            self.write_to_csv("tradeABCacceptC", proba_C_accept)
            self.write_to_csv("tradeABCrepB", rep_B)
            self.write_to_csv("tradeABCrepC", rep_C)
            self.write_to_csv("tradeABCactiontaken", actions)

    @staticmethod
    def write_to_csv(filename, variable):
        with open('CSV/'+filename+'.csv', 'a') as fd:
            writer = csv.writer(fd, quoting=csv.QUOTE_ALL)
            writer.writerow(variable)
