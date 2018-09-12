#!/usr/bin/env python
# -*- coding: utf-8 -*
import sys
import termios
import gym
from gym import envs

def getKey():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] &= ~termios.ICANON
    new[3] &= ~termios.ECHO

    try:
        termios.tcsetattr(fd, termios.TCSANOW, new)
        ch = sys.stdin.read(1)

    finally:
        termios.tcsetattr(fd, termios.TCSANOW, old)
    return ch

if __name__=='__main__':
    env = gym.make('CartPole-v0')
    observation = env.reset()
    action = 0
    for i in range(2000):
        env.render()
        key = getKey()
        if key == 'a':
            print "accelerate to the left"
            action = 0
        elif key == 'd':
            print "accelerate to the right"
            action = 1
        else:
            print "invalid key"
            exit()

        env.step(action)
