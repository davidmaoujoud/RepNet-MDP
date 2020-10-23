from Agent import Agent


class Oracle(Agent):
    """Oracle of the 2-agent trading experiment, extends the generic Agent class. The oracle is used to
    elicit certain reactions from the RepNet agent. The oracle first refuses all trade offers for 20
    time-steps, then accepts all trade offers for the subsequent 60 time-steps, before finally refusing
    all trade offers for the final 20 time-steps."""

    def __init__(self, g):
        Agent.__init__(self)
        self.s = 0
        self.previous_s = 0
        self.nb_steps = 0
        self.proba_selfish = 0
        self.g = g

    def lookahead(self, s):
        self.nb_steps += 1
        if self.nb_steps <= 20 or self.nb_steps >= 80:
            return 4  # Refuse
        else:
            return 3  # Accept

    def update(self, sp):
        self.s = sp
