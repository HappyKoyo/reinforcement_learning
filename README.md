# Rainforcement Learning  
This repository is for studying rainforcement learning.
I execute this program in following environment.

```
4.15.0-36-generic
Python 2.7.12
DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=16.04
DISTRIB_CODENAME=xenial
DISTRIB_DESCRIPTION="Ubuntu 16.04.4 LTS"
```

## cartpole_manual.py  
This program is playing cartpole in manual.  
You can know how difficult this game is.  
  
```
How to use
$ python manual_cartpole.py  
in terminal,
press -a- the cart move left  
press -d- the cart move right  
```

## cartpole_ql.py  
This program is playing cartpole using q-learning.  

```
How to use  
$ python cartpole_ql.py  
you can see learned motion in cartpole.
```

## cartpole_dqn.py
This program is playing cartpole using deep-q-network. You can also select double deep-q-network.

```
How to use
$ python cartpole_dqn.py
```

If you need to use ddqn, you have to change following line in cartpole_dqn.py.  

```
DQN_MODE = 0    # 0:DDQN 1:DQN
```

If program operated correctly, you can see the reward mean as following.  

![SuccessDQN](https://github.com/HappyKoyo/reinforcement_learning/tree/master/result_images/dqn_success.jpg "success")

Sometime (about once to 10 times) learning is failed as following.  

![FailedDQN](https://github.com/HappyKoyo/reinforcement_learning/tree/master/result_images/dqn_failed.jpg "failed")
