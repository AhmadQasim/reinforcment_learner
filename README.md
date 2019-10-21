# reinforcemnet_learner

Framework for training and testing reinforcement learning agents.



# Inventory Manager API

InventoryManagerEnv contains a producer model, a consumer model and the inventory.


## ProductItem
**ProductItem(item_type, production_time, expire_time)**

return an instance of product item

### Attributes
- age: age of the product item

### Methods
- is_done() -> boolean : return status of whether the product item is ready or not
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

### Attributes
- production_queue: a list of ProductItem list([ProductItem[type,age]])

### Methods

**is_busy() -> boolean**

return status of whether the producer model is producing or not

**start_producing(product_type, num_product) -> boolean**

Start producing. If the producer is currently not available, i.e. production_queue is not empty, discard production requests from the agent. Otherwise, add corresponding amount of product items into production queue.

- product_type: type of product to produce
- num_product: number of product to produce
- return: if request is accepted

**step() -> ReadyQueue**

1. aging of product items in the production queue

2. return ready queue (when products are ready: ready queue; not ready: empty list)

**reset() -> ProductionQueue**

clear production queue and return production queue

**get_state() -> ProducerState**

return production queue

**_is_busy()**

**_is_all_ready()**

## Consumer Model
Init(config[productlist, production_time, expire_time])

* get_order_queue() -> ConsumerState ([order_queue[Order[type, waiting_time]]])
* make_orders(inventory_products) -> list[Order]
* _serve_orders(inventory_products, make_orders) -> ConsumerState(OrderQueue), list[DoneOrder]
* step() -> ConsumerState "update orders"
* reset() -> ConsumerState
* get_state() -> ConsumerState


## Inventory
Init(config)

* reset()
* seed()
* step()
* render()