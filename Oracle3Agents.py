from Agent import Agent


class Oracle(Agent):
    """Oracle of the 3-agent trading experiment, extends the generic Agent class. The oracle is used to
    elicit certain reactions from the RepNet agent. In particular, the oracle allows for the setup of an
    experiment in which the RepNet agent is found to be unable to learn in circumstances in which it is
    not a primary actor."""

    def __init__(self, g):
        Agent.__init__(self)
        self.s = 0
        self.previous_s = 0
        self.nb_steps = 0
        self.proba_selfish = 0
        self.g = g

    def lookahead(self, s):
        self.nb_steps += 1
        if self.g == 1: # Agent B
            if self.nb_steps <= 33: # refuse trade offers
                return 1
            elif self.nb_steps <= 66:
                if self.s in [0]: # initial state, buy from A
                    return 1
                else: # accept trade offers
                    return 0
            else:
                return 1 # Trade with C and accept offers
        elif self.g == 2: # Agent C
            if self.nb_steps <= 33: # accept trade offers
                return 0
            elif self.nb_steps <= 66:
                if self.s in [0]: # initial state, buy from A
                    return 0
                else: # refuse trade offers
                    return 1
            else: # Trade with C and refuse trade offers
                return 0

        else:
            return 0

    def update(self, sp):
        self.s = sp
