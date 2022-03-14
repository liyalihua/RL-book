import numpy as np
from dataclasses import dataclass
import itertools
import pandas as pd
# import matplotlib.pyplot as plt

@dataclass
class Process1:
    '''mean reverting behavior of the stock price'''
    @dataclass
    class State:
        price: int
        '''Process1 contains two attributes level_param as int and alpha1 as float'''
    level_param: int # level to which price mean-reverts
    alpha1: float = 0.25 # strength of mean-reversion (non-negative value)
    def up_prob(self, state: State) -> float:
        return 1. / (1 + np.exp(-self.alpha1 * (self.level_param - state.price))) # prob. up movement as next state 
    def next_state(self, state: State) -> State:
        up_move: int = np.random.binomial(1, self.up_prob(state), 1)[0] # (n=1,p=prob,size=1) n as the set of outcome
        return Process1.State(price=state.price + up_move * 2 - 1)

def simulation(process, start_state):
    '''take process and start_state: compute the process.next_state )'''
    state = start_state
    while True:
        yield state
        state = process.next_state(state)

# a = simulation(Process1(level_param=120),start_state=100)

def process1_price_traces(
    start_price: int,level_param: int,
    alpha1: float,time_steps: int, num_traces: int
    ) -> np.ndarray:
    '''input dim and output dim'''
    process = Process1(level_param=level_param, alpha1=alpha1)
    start_state = Process1.State(price=start_price)
    return np.vstack([
        np.fromiter((s.price for s in itertools.islice(
        simulation(process, start_state),
        time_steps + 1
        )), float) for _ in range(num_traces)])

b = process1_price_traces(start_price = 100, level_param=70, alpha1 = 0.25,time_steps=100,num_traces=10)
print(b)
