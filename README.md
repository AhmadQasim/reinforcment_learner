# reinforcemnet_learner

Framework for training and testing reinforcement learning agents.



# Inventory Manager API

InventoryManagerEnv contains a producer model, a consumer model and the inventory.


# Reward
- minimze waiting time
- maximize freshnes
- minimize waste


## ProductItem
**ProductItem(item_type, production_time, expire_time)**

return an instance of product item

### Attributes
- age: age of the product item

### Methods
- is_done() -> boolean: return status of whether the product item is ready or not
- is_fresh() -> boolean: return status of freshment


## Order
**Order(item_type)**

return an instance of order

### Attributes
- waiting_time: waiting time since this order has been made

### Methods


## Producer Model
ProducerModel receives action from agent and schedules the production. It returns products when the production is ready. The state space of ProducerModel is a production queue of ProductItem.

**ProducerModel(config)**

Initialize a ProducerModel from config

- config: products list with prodution time and expire time for each individual product type

### Methods

**production_queue() -> ProductionQueue**

- production_queue: a list of ProductItem: list([ProductItem[type,age]])

**get_state() -> ProducerState**

return production queue

**start_producing(product_type, num_product) -> boolean**

Start producing. If the producer is currently not available, i.e. production_queue is not empty, discard production requests from the agent. Otherwise, add corresponding amount of product items into production queue.

- product_type: type of product to produce
- num_product: number of product to produce
- return: whether request is accepted or not

**step() -> ReadyQueue**

1. aging of product items in the production queue

2. return ready queue (when products are ready: ready queue; not ready: empty list)

**reset() -> ProductionQueue**

clear production queue and return production queue

**_is_busy() -> boolean**

return status of whether the producer model is producing or not

**_is_all_ready() -> boolean**

return status of whether all the products in the production queue are ready or not

## Consumer Model

**ConsumerModel(config)**

Initialize a ProducerModel from config
- config: products list with prodution time and expire time for each individual product type

### Methods

**order_queue() -> OrderQueue**

- consumer state: a list of order instances

**get_state() -> OrderQueue**

return state of consumer model. i.e. OrderQueue

**make_orders( inventory_products ) -> new_orders**

return new orders

**step() -> None**

add waiting time of orders

**reset() -> OrderQueue**

clear order queue and return order queue

**_server_orders( inventory_products ) -> ConsumerQueue, OrderQueue**

- consumerQueue: products that are ready to take
- orderQueue: pending orders


## Inventory

Inventory to keep track of products

### Methods

**reset() -> InventoryState**

- InventoryState: list of products

**step() -> None**

- aging of products in the inventory

**take(products) -> None**

- take products from the inventory

**add(products) -> None**

- add products into the inventory

## InventoryManagerEnv

gym environments for inventory managing


## Example Inventory Manager Environment
```
env = gym.make('gym_baking:Inventory-v0') # load environment

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