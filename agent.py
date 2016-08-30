import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.q = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
        self.alpha = 1.
        self.epsilon = 1.
        self.gamma = 0.5
        self.sr = 0
        self.random_state = np.random.RandomState(1999)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.q = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]];
        self.alpha = 1.
        self.epsilon = 1.
        self.gamma = 0.5

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)


        # Select current state
        states = ['green','restricted-green','red','expanded-red']
        state = -1

        if inputs['light'] == 'red':
            if inputs['left'] == None:
                state = 3
            else:
                state = 2
        elif inputs['light'] == 'green':
            if inputs['oncoming'] == None or inputs['oncoming'] == 'left':
                state = 0
            else:
                state = 1

        self.state = states[state]

        ## Select action according to your policy
        # reset action value
        action = 0
        # Decay Epsilon
        self.epsilon = self.epsilon*0.75
        # Explore vs. Learn based on Epsilon value
        rn = self.random_state.rand()
        if rn < self.epsilon:
            action = random.randint(0,3)
        else:
            if np.sum(self.q[state]) > 0:
                action = self.q[state].index(max(self.q[state]))
            else:
                action = random.randint(0,3)

        ## Execute action and get reward
        reward = self.env.act(self, self.env.valid_actions[action])

        ## Learn policy based on state, action, reward
        # Decay alpha
        self.alpha = self.alpha*0.4
        # Update Q
        update_q(self,state,action,reward,self.gamma,self.alpha)

        ## Success Rate
        if self.env.agent_states[self]['destination'] == self.env.agent_states[self]['location']:
            self.sr = self.sr + 1

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, state = {}".format(deadline, inputs, action, reward, self.state)  # [debug]

## Implementation of Q-Learning algorithm
def update_q(self,state, action, reward, alpha, gamma):
    new_q = alpha * (reward + gamma * max(get_all_states_values(self.q,state,action)))
    self.q[state][action] = new_q
    return self.q[state][action]

## Get values of all possible actions
def get_all_states_values(g,state,action):
    values = []
    for i in range(0,len(g)):
        for j in range(0,len(g[0])):
            if i != state and j != action:
                values.append(g[state][action])
    return values


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    print "Success Rate: " + str(a.sr)


if __name__ == '__main__':
    run()
