import uuid


class ProductItem:
    def __init__(self, item_type, production_time, expire_time):
        self._id = uuid.uuid1()
        self._item_type = item_type
        # how long it takes to produce
        self._production_time = production_time

        # how long the product stays fresh
        self._expire_time = expire_time

        # age of product, if negative it is still being produced
        self.age = -production_time

    def get_item_type(self):
        return self._item_type

    def is_done(self):
        return self.age > 0

    def is_fresh(self):
        return self.age < self._expire_time

    def get_age(self):
        return self.age

    def step(self):
        self.age += 1
