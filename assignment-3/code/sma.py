from queue import Queue

class SMA:
    """Simple moving average with incremental update."""
    def __init__(self, k):
        self.q = Queue(k)
        self.k = k
        self.SMA = 0

    def put(self, p):
        if self.q.full():
            self.SMA += (p - self.q.get()) / self.k
        else:
            self.SMA += p / self.k
        self.q.put(p)

    def get(self):
        return self.SMA

