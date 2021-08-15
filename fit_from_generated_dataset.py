import gym
import d3rlpy
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--env", type=str, default="CartPole-v1",
                    help="gym environment")
parser.add_argument("-c", "--collect-type", type=str, default="random", choices=["random", "training"],
                    help="gym environment")
args = parser.parse_args()

env = gym.make(args.env)

dataset = None
if args.collect_type == "random":
    # collect with random policy
    random_policy = d3rlpy.algos.DiscreteRandomPolicy()
    random_buffer = d3rlpy.online.buffers.ReplayBuffer(100000, env=env)
    random_policy.collect(env, buffer=random_buffer, n_steps=100000)
    dataset = random_buffer.to_mdp_dataset()
elif args.collect_type == "training":
    # collect during training
    sac = d3rlpy.algos.DiscreteSAC()
    replay_buffer = d3rlpy.online.buffers.ReplayBuffer(100000, env=env)
    sac.fit_online(env, buffer=replay_buffer, n_steps=100000)
    dataset = replay_buffer.to_mdp_dataset()

# setup CQL algorithm
cql = d3rlpy.algos.DiscreteCQL(use_gpu=False)

# split train and test episodes
train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

# start training
cql.fit(train_episodes,
        eval_episodes=test_episodes,
        n_epochs=1,
        experiment_name=args.collect_type + "_DiscreteCQL",
        scorers={
            'environment': d3rlpy.metrics.scorer.evaluate_on_environment(env),  # evaluate with CartPol-v0 environment
        })