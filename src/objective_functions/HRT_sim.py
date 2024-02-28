import numpy as np
from typing import List
from collections import deque


class HRTSim:
    def __init__(self, plog:List, tcm_size:int, l1_size:List, timescale:float, l1_latency:float, ram_latency:float) -> None:
        self.plog = plog
        self.tcm_size = tcm_size
        self.l1_size = l1_size
        self.ts = timescale
        
        self.l1_latency = l1_latency
        self.ram_latency = ram_latency

        self.symbol_list = list(set(self.plog))
        self._setup()

    def __call__(self, permutation: np.ndarray):
        hrt = 0
        self._populate_cache(permutation)
        for symbol in self.plog:
            if symbol in self.tcm:
                continue
            elif symbol in self.l1:
                self._update_symbol_usage(symbol)
                hrt += self.l1_latency
            else:
                hrt += self.ram_latency
                self._execute_eviction_policy(symbol)

        self._empty_cache()
        return hrt

    def _setup(self):
        self._empty_cache()

    def _update_symbol_usage(self, symbol):
        self.LRU_deque.remove(symbol)
        self.LRU_deque.append(symbol)

    def _execute_eviction_policy(self, symbol):
        # Type: Least Recently Used
        evicted_symbol = self.LRU_deque.popleft()
        self.LRU_deque.append(symbol)

        self.l1[self.l1.index(evicted_symbol)] = symbol

        self.ram.remove(symbol)
        self.ram.append(evicted_symbol)

    def _populate_cache(self, permutation):
        symbol_iter = iter(permutation)
        try:
            for i in range(self.tcm_size):
                self.tcm.append(next(symbol_iter))
            for i in range(self.l1_size):
                self.l1.append(next(symbol_iter))
            for i in range(int(1e6)):
                self.ram.append(next(symbol_iter))
        except StopIteration:
            pass

        self.LRU_deque=deque(self.l1)
    
    def _empty_cache(self):
        self.tcm =  deque([], maxlen=self.tcm_size) 
        self.l1 = deque([], maxlen=self.l1_size)
        self.ram = deque([], maxlen=int(1e6)) 
        self.LRU_deque = deque([])
        

if __name__ == "__main__":
    symbols = np.array(range(100))
    plog = np.random.choice(symbols, 10000)
    sim = HRTSim(plog, 10, 20, 1, 0.1, 1.5)
    permutation = np.array(range(100))
    np.random.shuffle(permutation)
    print(sim(permutation))