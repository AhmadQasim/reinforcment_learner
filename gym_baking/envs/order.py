class Order():
    def __init__(self, item_type):
        self._item_type = item_type
        self.is_done = False
        self.waiting_time = 0

    def step(self):
        if not self.is_done:
            self.waiting_time += 1