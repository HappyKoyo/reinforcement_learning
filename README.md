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

![SuccessDQN](https://user-images.githubusercontent.com/23475978/46568978-f5b7d180-c988-11e8-9550-236f5f3d388d.jpg "success")

Sometime (about once to 10 times) learning is failed as following.  

![FailedDQN](https://user-images.githubusercontent.com/23475978/46568995-4e876a00-c989-11e8-9dbf-7e913197c4fb.jpg "failed")
