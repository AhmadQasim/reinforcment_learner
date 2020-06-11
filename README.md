# reinforcement_learner

Framework for training and testing reinforcement learning agents.

# Installation
Install the inventory environment and dependencies using the following commands

```
$ git clone https://gitlab.virtualbaker.com/MachineLearning/reinforcement_learner.git
$ cd reinforcement_learner/
$ pip install -e .
```

# Introduction

This repository has the following structure
```
- root
    |- agents
    |- gym_baking
    |- result
    demo.py
    setup.py
```
Customer environments are defined in the "gym_baking" directory. To use it in your experiment, firstly you need to install this package and then call 
```
gym.make("gym_baking:YourCustomerEnvironment", **kwargs)
```
"agents" directory contains a random agent and a cem-agent to test the functionalities of the environment. You can try the cem-agent with inventory manager environment by
```
python agents/cem.py gym_baking:Inventory-v0 --display
```

"result" directory shows results of experiments on cem-agent in this inventory basic environment

 "demo.py" shows the dynamics of any environment interacting with a random agent
 ```
 python demo.py -e gym_baking:Inventory-v0
 ```


# Inventory Manager API

Inventory Manager (**InventoryManagerEnv**) is a gym environment where an agent can interact and learn from the reward. It mainly consists of a producer model (**ProducerModel**), a consumer model (**ConsumerModel**) and the inventory (**Inventory**). The metric and reward (**Metric**) is also included in the env. You will find the API of
- Metirc
- ProductItem
- Order
- Inventory
- ProducerModel
- ConsumerModel
- InventoryManagerEnv

in the next sections

# Metric and reward

- maximize sales (finished orders)
- minimize waste (not sold products end of day)

Orders can be a function of:

- time
- product freshness
- product availability
- customer waiting time

depending on consumer model.

**class Metric(config)**

Initialize an instance of Metric from config. Config is a dictionary contains products indices and attributes. It's parsed from the InventoryManagerEnv. You can find the example of config below.

**get_metric(state_history, done) -> reward, metric_info**

- state_history(dict) : state history from InventoryManagerEnv, should contains historical states of consumer, producer and inventory
- done(bool) : if this episode is done

return

- reward(int): 0 if episode is not done, otherwise reward = f(sales_ratio, products_ratio)

- info(dict): {} if episode is not done, otherwise the detailed metric stored in a dictionary
    - sales(int) : sum of all finished orders
    - waste(int) : sum of all products in the inventory
    - waits(int) : sum of all orders that are not finished
    - products(int) : sum of all produced products
    - sales_ratio : sales / (sales + waits)
    - products_ratio : products / (products + waste)

## ProductItem
**class ProductItem(item_type, production_time, expire_time)**

ProductItem is the basic element for one product that will be produced, stored and consumed. For example, if the the model decides to product a certain amount of products, then that amount of ProductItem should be instantiated and added to the production queue. 

Initilization:
- item_type: type of product
- production_time: how long it takes to produce
- expire_time: how long the product stays fresh

It has the following attributes:
- item_type: type of product
- age: age of the product item
- uuid: uuid of each product item

and methods:
- is_done() -> boolean: return status of whether the product item is ready or not
- is_fresh() -> boolean: return status of freshment
- step() -> None: aging of product item


## Order
**class Order(item_type)**

Order is the basic element for one order item made by consumer. It records the product type and waiting time for one order item. Similar to ProductItem, if the model makes a request for certain amount of products, then that amount of Order should be instantiated and added to the order queue.

Initialization:
- item_type : type of product
- waiting_time: waiting time since this order has made

Methods:
- step()-> None: aging of waiting time


## Producer Model
ProducerModel receives action from agent and schedules the production.

**class ProducerModel(config)**

Initialize a ProducerModel from config

- config(dict): products list with prodution time and expire time for each individual product type

### Methods

**get_state() -> producer_state**

return current states of the producer model. The state space of ProducerModel is a production queue of ProductItem.

- producer_state : dict
    - "production_queue" : list([ProductItem[type,age]])
    - "is_busy" : boolean # if oven is busy

**is_busy() -> boolean**

return status of whether the producer model is producing or not

**_is_all_ready() -> boolean**

return status of whether all the products in the production queue are ready or not

**get_ready_products() -> ready_products**

return ready queue (when products are ready: ready queue; not ready: empty list)

**start_producing(product_type, num_product) -> boolean**

Start producing. If the producer is currently not available, i.e. production_queue is not empty, discard production requests from the agent. Otherwise, add corresponding amount of product items into production queue.

- product_type (int): type of product to produce # index of product
- num_product (int): number of product to produce

Return
- boolean: whether request is accepted or not

**step() -> None**

aging of product items in the production queue

**reset() -> producer_state**

clear production queue and return the producer state

## Consumer Model
The consumer model makes orders based on the information of currently available inventory products, the entire products list, timestep, and current state of order queue. The base model has a private function called "_serve_order" to add new orders into order queue, compare order queue with inventory products, and return a serve queue which contains products that are ready to be taken from the inventory. Users should model new orders by "make_order" method of the consumer model.

**ConsumerModel(config)**

Initialize a ProducerModel from config
- config: products list with prodution time and expire time for each individual product type

### Methods

**get_state() -> consumer_state**

- consumer_state: dict
    - "order_queue" : list([Order[item_type]])

**make_orders( inventory_products, order_queue, time ) -> new_orders**

Return new orders: num_of_orders, list(item_type))

**_server_orders( inventory_products, time ) -> serve_queue**

Add new orders into order queue. Then split available products and not available products based on the comparation between current order queue and the inventory products

Return
- serve_queue: products that are ready to be taken from the inventory

**step() -> None**

add waiting time of orders

**reset() -> consumer_state**

clear order queue and return consumer status



## Inventory

Inventory keeps track of products that are newly produced and taken from consumer.

### Methods

**get_state() -> inventory_state**

- inventory_state : dict
    - "products" : list(ProductItem) # products in the inventory

**reset() -> inventory_state**

clear products list and return state

**step() -> None**

- aging of products in the inventory

**take(products) -> None**

take products from the inventory
- products(list): a list of ProductItem that will be taken from the inventory

**add(products) -> None**

add products into the inventory

-products(list): a list of ProductItem that will be added into the inventory. The user is responsible to make sure that all products are available in the inventory.


## InventoryManagerEnv

InventoryManagerEnv is an environment in gym. It captures inventory dynamics through the managment of producer model, consumer model and inventory.  As a gym environment object, it performs step transformation and return reward. It has basic functions including step, render, reset and seed.

**class InventoryManagerEnv(config_path)**

Initialize a Inventory Manager Environment from a config.yaml file. The config file are parsed into a dictionary. Then the dictionary is passed into producer, consumer, inventory and metric

**action space**
```
R^2
```

- [0, num_products]
- [0, +inf]

**observation space**
- production_queue list(ProductItem)
- inventory_products list(ProductItem)
- consumer_queue list(Order)

```
gym.make("gym_baking:Inventory-v0", config_path = /path/to/config_file)
...

class InventoryManagerEnv():
    def __init__(self, config_path)
        config = yaml.load(config_path)
        self._producer_model = ProducerModel(config)
        self._consumer_model = ConsumerModel(config)
        self._inventory = Inventory(config)
        self._metric = Metric(config)
```

### Example Config
```
title: "Inventory Manager Environment"
description: "Config for inventory manager environment"

product_list:
  0:
    type: brot
    production_time: 5
    expire_time: 100
  
  1:
    type: pretzel
    production_time: 15
    expire_time: 20

episode_max_steps: 100
```

**reset() -> observation**

reset the environment and return observation

**seed() -> seed**

add seed for environment

**render() -> image**

plottings of some entities

return
- image(numpy.ndarray): numpy image of the current figure. Useful for gym.wrappers.Monitor

**step(action) -> observation, reward, done, info**

Run one timestep of the environment

### example step
```
def step(self,action):
    ready_products = self._producer_model.get_ready_products()
    self._inventory.add(ready_products)
    curr_products = self._inventory.get_state()["products"]
    consumption_products = self._consumer_model._serve_orders(curr_products)
    self._inventory.take(consumption_products)
    self._producer_model.start_producing(action)
    self._producer_model.step()
    self._consumer_model.step()
    self._inventory.step()

    states = get_states()
    state_history = get_state_history()
    observations = {key:states[key] for key in ["producer_state", "consumer_state", "inventory_state"]}
    done = episode_is_done()
    reward, metric_info = self._metric.get_metric(state_history, done)
    
    return observations, reward, done, metric_info
```


## Example Inventory Manager Environment
```
env = gym.make('gym_baking:Inventory-v0', config_path="inventory.yaml") # load environment

for episode in num_episodes:
    observation = env.reset() # reset environment
    reward = 0
    done = False
    for step in num_timesteps:
        env.render() # visualization
        action = agent.act(observation, reward, done) # agent makes desicion
        observation, reward, done, info = env.step(action) # environment transformation
        if done:
            break
```

## Example Random Agent
```
class Agent(object):
    def __init__(self, action_space):
        self.action_space = action_space
    
    def act(self, observation, reward, done):
        is_busy = observation["producer_model"]["is_busy"]
        action = f(observation, self.action_space)
        return action
```