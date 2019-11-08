from gym_baking.envs.order import Order
from gym_baking.envs.product_item import ProductItem
from typing import List
from collections import Counter


def unwrap_order(order_queue: List[Order]):
    return [(x.get_item_type(), x.get_waiting_time()) for x in order_queue]


def unwrap_product_item(production_queue: List[ProductItem]):
    return [(x.get_item_type(), x.is_fresh()) for x in production_queue]


def observation_state_vector(observation, return_count = False):
    production_queue = unwrap_product_item(observation['producer_state']['production_queue'])
    inventory = observation['inventory_state']['products']
    order_queue = unwrap_order(observation['consumer_state']['order_queue'])

    if return_count:
        return [dict(Counter(item[0] for item in production_queue)),
                dict(Counter(item[0] for item in inventory)),
                dict(Counter(item[0] for item in order_queue))]
    else:
        return [production_queue, inventory, order_queue]
