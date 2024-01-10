"""
Test the text gym environment.

python ./run_envs/run_web_agent_text_env.py --policy human

TODO: move to testing dir for more rigorous tests
"""

#!/bin/bash

import sys,os
sys.path.append(os.getcwd())

import gym
from rich import print
from rich.markup import escape

from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.models import RandomPolicy,HumanPolicy
from web_agent_site.utils import DEBUG_PROD_SIZE

import time,argparse
parser = argparse.ArgumentParser()
parser.add_argument("--prods", default= None, type=int)
parser.add_argument("--policy", default= "random", type=str, choices = ['human', 'random'])
args = parser.parse_args()

DEBUG_PROD_SIZE = args.prods

if __name__ == '__main__':
    env = gym.make('WebAgentTextEnv-v0', observation_mode='text', num_products=DEBUG_PROD_SIZE)
    env.reset()
    
    ii = 0
    try:
        if args.policy == 'random':
            policy = RandomPolicy()
        elif args.policy == 'human':
            policy = HumanPolicy()
    
        observation = env.observation
        while True:
            print("observation===>", observation)
            print()

            available_actions = env.get_available_actions()
            print('Available actions:', available_actions)
            print()

            action = policy.forward(observation, available_actions)
            observation, reward, done, info = env.step(action)
            print(f'Taking action "{escape(action)}" -> Reward = {reward}')
            print('---'*30 + '\n')

            if done:
                break
            ii += 1
    finally:
        env.close()