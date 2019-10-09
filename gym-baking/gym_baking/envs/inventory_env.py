import uuid

class ProductItem():
    def __init__(self, item_type, production_time, expire_time):
        self._id = uuid.uuid1()
        self._item_type = item_type
        # how long it takes to produce
        self._production_time = production_time

        # how long the product stays fresh
        self._expire_time = expire_time

        # age of product, if negative it is still being produced
        self._age = -production_time
    
    def is_done(self):
        return self.age > 0

    def is_fresh(self):
        return self.age < self._expire_time

    def step(self):
        self.age += 1


class Inventory():
    def __init__(self):
        self._products = {}

    def add(self, product):
        self._products[product._id] = product
        
    def take(self, product_id):
        return self._products.pop(product_id)
    
    def products(self):
        return self._products

    def step(self):
        # update items
        for item in self._products.items():
            item.step()

class ProducerModel():
    def __init__():
        self._production_queue()

    
    def start_producing(self, products):


    def step():