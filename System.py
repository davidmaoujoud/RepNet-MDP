from Experiments import MDPParameters


class System:
    """Represents the System tuple. This class contains the information of interest and available to
    all agents in the network."""

    def __init__(self, parameters: MDPParameters.MDPParameters):
        self.parameters = parameters
        self.agents = parameters.agents
        self.states = parameters.states
        self.actions_u = parameters.actions_u
        self.actions_d = parameters.actions_d
        self.actions = self.actions_u + self.actions_d
        self.current_state = self.states[parameters.first_state]
        self.impact = parameters.impact_function

    def I(self, h, i, s, a_i):
        """
        Impact function, returns the impact on agent h that is due to agent i performing action a_i
        in environment state s
        """
        return self.impact[h][i][s][a_i]

    def U(self, current_image_level, current_impact):
        """
        Image update function, returns an updated image level given current_image_level and new
        impact current_impact
        """
        if self.parameters.update_function == "difference_update":
            if current_impact >= 0:
                return current_image_level + self.parameters.learning_rate * (1 - current_image_level) * current_impact
            else:
                return current_image_level + self.parameters.learning_rate * (1 + current_image_level) * current_impact

        elif self.parameters.update_function == "saturation_update":
            if current_image_level + self.parameters.learning_rate * current_impact > 1:
                return 1
            elif current_image_level + self.parameters.learning_rate * current_impact < -1:
                return -1
            else:
                return current_image_level + self.parameters.learning_rate * current_impact

        else:
            return 0


