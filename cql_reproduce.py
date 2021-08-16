import gym
import d3rlpy
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gpu", action="store_true",
                    help="use GPU")
parser.add_argument("--epochs", type=int, default=3000,
                    help="num of epochs. default is as same as "
                         "https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_mujoco_new.py")
parser.add_argument("-e", "--env", type=str, default="hopper-medium-v0",
                    help="gym environment")
args = parser.parse_args()

dataset, env = d3rlpy.datasets.get_dataset(args.env)

# setup CQL algorithm
cql = d3rlpy.algos.CQL(use_gpu=args.gpu)

# split train and test episodes
train_episodes, test_episodes = train_test_split(dataset, test_size=0.2)

# start training
cql.fit(train_episodes,
        eval_episodes=test_episodes,
        n_epochs=args.epochs,
        experiment_name="CQL_epochs" + str(args.epochs),
        scorers={
            'environment': d3rlpy.metrics.scorer.evaluate_on_environment(env),  # evaluate with CartPol-v0 environment
        })
