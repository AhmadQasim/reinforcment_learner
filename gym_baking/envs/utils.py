import numpy as np
from gym_baking.envs.order import Order
from gym_baking.envs.product_item import ProductItem
from typing import List
from collections import Counter


def unwrap_order(order_queue: List[Order], items_to_id):
    return np.array([(items_to_id[x.get_item_type()], x.get_waiting_time()) for x in order_queue])


def unwrap_product_item(production_queue: List[ProductItem], items_to_id):
    return np.array([(items_to_id[x.get_item_type()], x.is_fresh()) for x in production_queue])


def unwrap_inventory_item(inventory_queue: List[ProductItem], items_to_id):
    return np.array([(items_to_id[x.get_item_type()], x.get_age()) for x in inventory_queue])


def prepare_dictionary(dictionary, items_to_id, source):
    means = np.zeros(shape=len(items_to_id.keys()))
    for key, val in dictionary.items():
        item_ages = source[source[:, 0] == key, 1]
        means[key] = np.mean(item_ages)
        dictionary[key] = [dictionary[key], int(means[key])]

    return dictionary


def observation_state_vector(observation, items_to_id, return_count = False):

    production_queue = unwrap_product_item(observation['producer_state']['production_queue'], items_to_id)
    inventory = unwrap_inventory_item(observation['inventory_state']['products'], items_to_id)
    order_queue = unwrap_order(observation['consumer_state']['order_queue'], items_to_id)

    inventory_dict = prepare_dictionary(dict(Counter(item[0] for item in inventory)), items_to_id, inventory)
    order_dict = prepare_dictionary(dict(Counter(item[0] for item in order_queue)), items_to_id, order_queue)

    if return_count:
        return [dict(Counter(item[0] for item in production_queue)),
                inventory_dict,
                order_dict]
    else:
        return [production_queue, inventory, order_queue]


def map_items_to_id(configs):
    mapping = dict()
    for key, val in configs['product_list'].items():
        mapping[val['type']] = key

    return mapping
