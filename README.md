# RepNet-MDP

This project contains the implementation of several planning and learning algorithms:
- RepNet-MDP: the framework of interest, it is a multi-agent extension to classic MDPs. The current implementation contains a RepNet algorithm that performs online planning.
- MDP: an implementation of the MDP framework that can perform online planning.
- Q-learner: an epsilon-greedy implementation of the Q-learning algorithm.
- Oracle: an algorithm that simply follows a specific set of rules defined by us.

In addition, the project consists of the means to run the two simple trading experiments described in the [RepNet-MDP paper](https://arxiv.org/abs/2008.11791):
- A trading example between 2 agents, see [Section 7.1 of the RepNet-MDP paper](https://arxiv.org/abs/2008.11791).
- A trading example between 3 agents, see [Section 7.2 of the RepNet-MDP paper](https://arxiv.org/abs/2008.11791).

## Running the experiments:
To run either experiment, you need to execute the ```main()``` function in ```Main.py```. ```Main.py``` contains the variable ```experiment_number```, which allows you to select the experiment you wish to run.
Furthermore, the parameter files of both experiments (```Trade2Agents.py``` and ```Trade3Agents.py```) can be found in the ```Experiments/``` directory, the parameters may be altered to fit different variations of the experiments:
- Basic RepNet/Q-learning parameters such as the **decay**, **learning rate**, and **epsilon** can be tuned.
- The depth **D** of the lookahead search tree (for RepNet and regular MDP agents) can be set. The greater the value for **D**, the slower the algorithm, but the more insightful the decision-making.
- The types of agents that make up the experiment can be set. For instance, the 3-agent trading scenario could consist of 3 RepNet agents, or 1 RepNet agent and 2 Q-learners (or any other combination of frameworks).
- The objective and subjective transition models **OT** and **ST** of each agent can be tweaked and will lead to the agents behaving differently.
- The RepNet impact function **I** (analogous to the MDP immediate reward) can be tweaked and will lead to the RepNet agents behaving differently.
- A different update function **U** can be selected. The current implementation contains two variants: the ```saturation update function``` and the ```difference update function```.

Different variables are tracked depending on the experiment. At the end of each run, the relevant results are stored in the **CSV** files that can be found in the ```CSV/``` directory.
