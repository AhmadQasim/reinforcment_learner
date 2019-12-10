class Order():
    def __init__(self, item_type):
        self._item_type = item_type
        self.is_done = False
        self.waiting_time = 0

    def get_item_type(self):
        return self._item_type

    def get_waiting_time(self):
        return self.waiting_time

    def step(self):
        if not self.is_done:
            self.waiting_time += 1