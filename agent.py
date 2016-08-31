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
        self.q = [];
        self.alpha = 0.9
        self.epsilon = 0.75
        self.gamma = 0.75
        self.sr = 0
        self.random_state = np.random.RandomState(1999)
        self.states = []

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.q = [];
        self.alpha = 0.9
        self.epsilon = 0.75
        self.gamma = 0.75
        self.states = []

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)


        # Select current state
        current_state = {'light' : inputs['light'], 'left' : inputs['left'], 'oncoming' : inputs['oncoming'], 'next' : self.next_waypoint}

        # if current state is not registered yet add it
        if current_state not in self.states:
            self.states.append(current_state)
            self.q.append([0,0,0,0])

        # get the index of the current state
        current_state_index = self.states.index(current_state)

        self.state = current_state

        ## Select action according to your policy
        # reset action value
        action = 0
        # Decay Epsilon
        self.epsilon = self.epsilon*0.25
        # Explore vs. Learn based on Epsilon value
        rn = self.random_state.rand()
        if rn < self.epsilon:
            action = random.randint(0,3)
        else:
            if np.sum(self.q[current_state_index]) > 0:
                action = self.q[current_state_index].index(max(self.q[current_state_index]))
            else:
                action = random.randint(0,3)

        ## Execute action and get reward
        reward = self.env.act(self, self.env.valid_actions[action])

        inputs = self.env.sense(self)

        # Select current state
        new_state = {'light' : inputs['light'], 'left' : inputs['left'], 'oncoming' : inputs['oncoming'], 'next' : self.next_waypoint}

        # if new state is not registered yet add it
        if new_state not in self.states:
            self.states.append(new_state)
            self.q.append([0,0,0,0])

        # get the index of the new state
        new_state_index = self.states.index(new_state)

        ## Learn policy based on state, action, reward
        # Decay alpha
        self.alpha = self.alpha*0.4
        # Update Q
        update_q(self,current_state_index,new_state_index,action,reward,self.gamma,self.alpha)

        ## Success Rate
        if self.env.agent_states[self]['destination'] == self.env.agent_states[self]['location']:
            self.sr = self.sr + 1

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, state = {}".format(deadline, inputs, action, reward, self.state)  # [debug]

## Implementation of Q-Learning algorithm
def update_q(self,current_state,new_state,action,reward,alpha, gamma):
    self.q[current_state][action] = self.q[current_state][action] + alpha * (reward + gamma * max(self.q[new_state]) - self.q[current_state][action])
    return self.q[current_state][action]

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
