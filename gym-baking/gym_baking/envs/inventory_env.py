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

class ProducerItem():
    def __init__(self, product, amount, usebatch):
        self._item = ProductItem(product)
        self._amount = amount
        self.usebatch = usebatch

class ConsumerItem():
    def __init__(self, product, amount, fresh_thresh):
        self._item = ProductItem()
        self._amount = amount
        self._fresh_thresh = fresh_thresh
        self.waiting = 0

    def step(self):
        self.waiting += 1

class Inventory():
    def __init__(self):
        self._products = {}
        self._wastes = []
        self._producer_model = ProducerModel()
        self._consumer_model = ConsumerModel()

    def add(self, product):
        self._products[product._id] = product
        
    def take(self, product_id):
        return self._products.pop(product_id)

    def discard(self):
        for product in self._products.items():
            if not product.is_fresh():
                waste = self._products.pop(product._id)
                self._waste_queue.push(waste)
    
    def products(self):
        return self._products

    def step(self):
        # update items
        for item in self._products.items():
            item.step()
        
        product = self._producer_model.step()
        self.add(product)

        product = self._consumer_model.step()
        self.take(product)

        self.discard()

        return self._products, self._wastes

class ProducerModel():
    def __init__(self, oven_max_volume, product_min_batch):
        self._oven_max_volume = oven_max_volume
        self._product_min_batch = product_min_batch

        # wait for oven ready or batch baking
        self._preparation_queue()

        # baking in the oven
        self._oven_queue()

        # product is ready
        self._production_queue()
        
    def is_oven_full(self):
        return len(self._oven_queue)== self._oven_max_volume

    def is_oven_empty(self):
        return len(self._oven_queue)==0
    
    def products_in_oven(self):
        return self._oven_queue()
    
    def start_producing(self, products):
        while not self.is_oven_full():
            product = products.pop()
            if not product["usebatch"]:
                self._oven_queue.push(product)
            elif product["usebatch"]:
                if self._preperation_queue.get(product._id["amount"], 0) >self._product_min_batch:
                    self._oven_queue.push(product)

        for product in products:
            self._preparation_queue.push(product)

    def update_product(self):
        for item in self._oven_queue():
            item.step()
        for product in self._oven_queue():
            if product.is_done():
                self._production_queue.push(product)
                self._oven_queue.pop(product)

    def step(self, products):
        """
        products: list[ProducerItem]  ProducerItem: (name, amount, usebatch)
        """
        self.update_product()
        self.start_producing(products)
        return self._production_queue()

        
class ConsumerModel():
    def __init__(self):
        self._consumer_queue = {}
        self._consumer_finished_list = list()

    def has_no_consumer(self):
        return len(self._consumer_queue)==0

    def is_waiting(self):
        return len(self._consumer_queue)>0

    def done_consume(self):
        return self._consumer_finished_list

    def consume(self):
        return self._consumer_queue()

    def place_order(self, products):
        for product in products:
            self._consumer_queue.push(product)

    def cancel_order(self, products):
        for product in products:
            self._consumer_queue.pop(product)

    def take(self):
        for product in self._consumer_queue:
            if product in self._production_queue and self._production_queue[product].is_fresh():
                self._production_queue.pop(product)
                self._consumer_queue.pop(product)

    def step(self, products, withdraw_products):
        """
        products: list[ConsumerItem] ConsumerItem: (product, amount, isfresh)
        """
        self.place_order(products)
        self.cancel_order(withdraw_products)
        return self.consume()