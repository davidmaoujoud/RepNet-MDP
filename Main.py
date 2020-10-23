import System
import OnlineSolver
import RepNetAgent
import Oracle2Agents
import Oracle3Agents
import MDPAgent
import QLearner
from Experiments import Trade2Agents
from Experiments import Trade3Agents


# Which experiment? 1 = Trade between two agents, 2 = Trade between 3 agents
experiment_number = 1


def main():
    if experiment_number == 1:
        parameters = Trade2Agents.Trade2Agents()
    elif experiment_number == 2:
        parameters = Trade3Agents.Trade3Agents()
    else: # Default
        parameters = Trade2Agents.Trade2Agents()

    system = System.System(parameters)
    agents = []

    for g in parameters.agents:
        if parameters.agent_types[g] == "repnet":
            agents.append(RepNetAgent.RepNetAgent(g, system, parameters))
        elif parameters.agent_types[g] == "oracle":
            if experiment_number == 1:
                agents.append(Oracle2Agents.Oracle(g))
            elif experiment_number == 2:
                agents.append(Oracle3Agents.Oracle(g))
            else: # Default
                agents.append(Oracle2Agents.Oracle(g))
        elif parameters.agent_types[g] == "mdp":
            agents.append(MDPAgent.MDPAgent(g, system, parameters))
        elif parameters.agent_types[g] == "qlearner":
            agents.append(QLearner.QLearner(g, system, parameters))
    repNetMDP = OnlineSolver.OnlineSolver(system, agents, parameters, experiment_number=experiment_number)

    repNetMDP.online_repnet_solver()


if __name__ == "__main__":
    main()
