class Inventory:
    def __init__(self):
        self._products = []
        self.state = {'products': self._products}

    def reset(self):
        self._products.clear()
        return self.get_state()

    def add(self, products):
        for item in products:
            self._products.append(item)

    def take(self, products):
        for item in products:
            self._products.remove(item)

    def get_state(self):
        self.state["products"] = self._products.copy()
        return self.state

    def step(self):
        for item in self._products:
            item.step()