from agents.threshold_agent import BaselineAgent
from agents.adp import DPAgent
import logging
import json
import os
import multiprocessing

MODE = "in-d"
TESTS = 10
JSON_PATH = "./reinforcemnet_learner/result"

base_results = []
dp_ar = []
dp_or = []


def worker(seed, base_dict, dp_ar_dict, dp_or_dict):
    # run baseline agent
    base_agent = BaselineAgent()
    base_dict[seed] = base_agent.test(seed=seed)

    # run DP+AR
    agent = DPAgent(config_path="./reinforcemnet_learner/inventory.yaml", loglevel=logging.CRITICAL)
    dp_ar_dict[seed] = agent.train_with_env(seed=seed)

    # run DP+OR
    # agent_test = DPAgent(config_path="./reinforcemnet_learner/inventory.yaml", loglevel=logging.CRITICAL)
    # dp_or_dict[seed] = agent_test.train_with_env(test_seed=seed)


jobs = []
manager = multiprocessing.Manager()
base_dict = manager.dict()
dp_ar_dict = manager.dict()
dp_or_dict = manager.dict()

for i in range(TESTS):
    p = multiprocessing.Process(target=worker, args=(i, base_dict, dp_ar_dict, dp_or_dict, ))
    jobs.append(p)
    p.start()

for i in range(TESTS):
    jobs[i].join()

print(base_dict, dp_ar_dict, dp_or_dict)

with open(os.path.join(JSON_PATH, MODE + ".json"), "w") as f:
    json.dump({"baseline": base_dict.copy(), "dp+ar": dp_ar_dict.copy(), "dp+or": dp_or_dict.copy()}, f, indent=4)




with open(os.path.join(JSON_PATH, MODE + ".json"), "r") as f:
    data = json.load(f)

baseline_sale_ratio = []
baseline_prod_ratio = []

dp_ar_sale_ratio = []
dp_ar_prod_ratio = []

dp_or_sale_ratio = []
dp_or_prod_ratio = []

print(data)

for i in range(TESTS):
    baseline_sale_ratio.append(data["baseline"][str(i)][1]["sale_miss_ratio"])
    baseline_prod_ratio.append(data["baseline"][str(i)][1]["product_wait_ratio"])

    dp_ar_sale_ratio.append(data["dp+ar"][str(i)][1]["sale_miss_ratio"])
    dp_ar_prod_ratio.append(data["dp+ar"][str(i)][1]["product_wait_ratio"])

    # dp_or_sale_ratio.append(data["dp+or"][str(i)][1]["sale_miss_ratio"])
    # dp_or_prod_ratio.append(data["dp+or"][str(i)][1]["product_wait_ratio"])

print(f"Baseline average sales ratio: {sum(baseline_sale_ratio)/len(baseline_sale_ratio)}")
print(f"Baseline average prod ratio: {sum(baseline_prod_ratio)/len(baseline_prod_ratio)}")

print(f"dp+ar average sales ratio: {sum(dp_ar_sale_ratio)/len(dp_ar_sale_ratio)}")
print(f"dp+ar average prod ratio: {sum(dp_ar_prod_ratio)/len(dp_ar_prod_ratio)}")

# print(f"dp+or average sales ratio: {sum(dp_or_sale_ratio)/len(dp_or_sale_ratio)}")
# print(f"dp+or average prod ratio: {sum(dp_or_prod_ratio)/len(dp_or_prod_ratio)}")