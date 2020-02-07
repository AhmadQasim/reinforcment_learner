from agents.adp import DPAgent
import yaml
from gym_baking.envs.consumers.parametric_consumer import PoissonConsumerModel as Oracle
from collections import Counter
import numpy as np

weights = [[10,25,15,10], [5,2,30,20]]
total_seeds = 10

result_dic = {}
with open("inventory.yaml", 'r') as f:
    config = yaml.load(f, Loader=yaml.Loader)
total_mean_waiting = 0
total_miss_ratio = 0
number_of_products = len(weights)

def vectorize_order(order):
    counter_obj = Counter(order[1])
    counts = vectorize_counter_np(counter_obj)
    return counts


def vectorize_counter_np(counter):
    counts = np.zeros(number_of_products, dtype="int64")
    for indx, count in counter.items():
        counts[indx] = count
    return counts


for seed in range(total_seeds):
    agent = DPAgent(config_path="inventory.yaml")
    consumer = Oracle(config)
    consumer.set_counts(weights)
    samples = [vectorize_order(tup) for tup in consumer.give_all_samples(seed)]

    agent.inject_prophecy(samples)
    agent.train()
    mean_waiting, miss_ratio = agent.print_optimal()
    total_mean_waiting += mean_waiting
    total_miss_ratio += miss_ratio
    result_dic[f"seed:{seed}"] = (f"miss_ratio: {miss_ratio}", f"mean_waiting: {mean_waiting}")

result_dic[f"average_miss_ratio_weights_{weights}"] = total_miss_ratio/total_seeds
result_dic[f"average_mean_waiting_weights_{weights}"] = total_mean_waiting/total_seeds

print(result_dic)
