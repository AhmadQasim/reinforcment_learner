# reinforcemnet_learner

Framework for training and testing reinforcement learning agents.



# Inventory Manager API

InventoryManagerEnv contains a producer model, a consumer model and the inventory.

# Metric and reward 

- maximize sales (finished orders)
- minimize waste (not sold products end of day)

Orders can be a function of:

- time
- product freshness
- product availability
- customer waiting time

depending on consumer model.

## ProductItem
**ProductItem(item_type, production_time, expire_time)**

return an instance of product item

### Attributes
- age: age of the product item
- uuid: uuid of each product item

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

**get_state() -> producer_state**

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

- product_type (int): type of product to produce # index of product ?? layout of config file
- num_product (int): number of product to produce

Return
- boolean: whether request is accepted or not

**step() -> None**

1. aging of product items in the production queue

**reset() -> producer_state**

clear production queue and return the producer state

## Consumer Model

**ConsumerModel(config)**

Initialize a ProducerModel from config
- config: products list with prodution time and expire time for each individual product type

### Methods

**get_state() -> consumer_state**

- consumer_state: dict
    - "order_queue" : list([Order[item_type]])

**make_orders( inventory_products, order_queue, time ) -> new_orders**

Return new orders: list(tuple(order_type, num_order))

**_server_orders( inventory_products, time ) -> taken_queue, order_queue**
Add new orders into order queue. Then split available products and not available products based on the comparation between current order queue and the inventory products

Return
- taken_queue: products that are ready to take
- order_queue: pending orders (order_queue) # want to remove it

**step() -> None**

add waiting time of orders

**reset() -> consumer_state**

clear order queue and return consumer status



## Inventory

Inventory to keep track of products

### Methods

**get_state() -> inventory_state**

- inventory_state : dict
    - "products" : list(ProductItem) # products in the inventory

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

**InventoryManagerEnv()**

Initialize a Inventory Manager Environment. Producer model, consumer model and inventory are defined here
```
self._producer_model = ProducerModel()
self._consumer_model = ConsumerModel()
self._inventory = Inventory()
```
### Example Config
```
        self.config = {
            0: {'type':'brot','production_time':5, 'expire_time':100},
            1: {'type':'pretzel','production_time':15, 'expire_time':20},
        }
```

**reset() -> observation**

reset environment and return observation

**seed() -> seed**

set seed

**render() -> image**

plotting function

return
- image(numpy.ndarray): numpy version image of the current figure. useful for gym.wrappers.Monitor

**step(action) -> observation, reward, done, info**

Run one timestep of the environment

### example step workflow
```
def step(self,action):
    ready_products = self._producer_model.get_ready_products()
    self._inventory.add(ready_products)
    curr_products = self._inventory.get_state()["products"]
    consumption_products, orderqueue = self._consumer_model._serve_orders(curr_products)
    self._inventory.take(consumption_products)
    self._producer_model.start_producing(action)
    self._producer_model.step()
    self._consumer_model.step()
    self._inventory.step()

    states = get_states()
    reward = reward_function()
    done = episode_is_done()
    
    return states, reward, done, {}
```


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