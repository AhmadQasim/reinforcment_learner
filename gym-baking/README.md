library of baking environments for gym


# Inventory API

## Producer Model
Init(config[productlist, production_time, expire_time])

* start_produce(product_type, num_product) -> None
* step() -> ProducerState (production_queue[ProductItem[type, age]])
* reset() -> ProducerState  "reset producer state"
* get_state() -> ProducerState
* get_ready_product() -> ReadyQueue (will be deprecated)
* is_busy()
* is_all_ready()

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