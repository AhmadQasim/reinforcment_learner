from __future__ import print_function

import gym
from gym import wrappers, logger
import numpy as np
from six.moves import cPickle as pickle
import json, sys, os
from os import path
import time
import matplotlib.pyplot as plt
# from agents._policies import BinaryActionLinearPolicy as LinearPolicy
from agents._policies import LinearActionLinearPolicy as LinearPolicy

import argparse
import gym_baking
from inventory_wrapper import InventoryQueueToVector

def cem(f, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function

    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    """
    n_elite = int(np.round(batch_size*elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        ths = np.array([th_mean + dth for dth in  th_std[None,:]*np.random.randn(batch_size, th_mean.size)])
        ys = np.array([f(th) for th in ths])
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.std(axis=0)
        yield {'ys' : ys, 'theta_mean' : th_mean, 'y_mean' : ys.mean(), 'theta_init':ths[0]}

def do_rollout(agent, env, num_steps, render=False, verbose=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        if render:
            env.render()
        a = agent.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if done: break
    if verbose:
        print(_info)
    if render:
        plt.savefig(str(int(time.time()*1000)) +'.jpg')
        env.close()
    return total_rew, t+1

if __name__ == '__main__':
    logger.set_level(logger.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('target', nargs="?", default="CartPole-v0")
    parser.add_argument('-t', '--num_iter', type=int, default=100)
    parser.add_argument('--display', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    env = gym.make(args.target)
    # env.seed(1)
    params = dict(n_iter=args.num_iter, batch_size=25, elite_frac=0.2)
    num_steps = 200

    # You provide the directory to write to (can be an existing
    # directory, but can't contain previous monitor results. You can
    # also dump to a tempdir if you'd like: tempfile.mkdtemp().
    outdir = 'cem-agent-results'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # env = wrappers.Monitor(env, outdir, force=True)
    if 'Inventory-v0' in args.target:
        env = InventoryQueueToVector(env)


    if args.test:
        model_path = 'cem_agent.npy'
        print('start test model {}'.format(model_path))
        theta = np.load(model_path)
        agent = LinearPolicy(theta)
        do_rollout(agent, env, 200, args.display, verbose=args.verbose)
        exit()

    # Prepare snapshotting
    # ----------------------------------------
    def writefile(fname, s):
        with open(path.join(outdir, fname), 'w') as fh: fh.write(s)
    info = {}
    info['params'] = params
    info['argv'] = sys.argv
    info['env_id'] = env.spec.id
    # ------------------------------------------

    def noisy_evaluation(theta):
        policy = LinearPolicy
        agent = policy(theta)
        rew, T = do_rollout(agent, env, num_steps)
        return rew

    # Train the agent, and snapshot each stage
    if 'Inventory-v0' in args.target:
        th_init = np.zeros(env.observation_space.shape[0])
    else:
        th_init = np.zeros(env.observation_space.shape[0]+1)

    for (i, iterdata) in enumerate(cem(noisy_evaluation, th_init, **params)):
        print('Iteration %2i. Episode mean reward: %7.3f'%(i, iterdata['y_mean']))
        if i%10==0:
            agent = LinearPolicy(iterdata['theta_mean'])
            do_rollout(agent, env, 200, render=args.display, verbose=args.verbose)


    agent = LinearPolicy(iterdata['theta_mean'])
    do_rollout(agent, env, 200, render=args.display, verbose=args.verbose)

    # Write out the env at the end so we store the parameters of this
    # environment.
    np.save('cem_agent.npy', iterdata['theta_mean'])
    writefile('info.json', json.dumps(info))

    env.close()
