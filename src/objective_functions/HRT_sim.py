import numpy as np
from typing import List
from tqdm import tqdm

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

    def __call__(self, permutation: List):
        hrt = 0
        self._populate_cache(permutation)
        for symbol in tqdm(self.plog):
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
        for item in self.l1:
            self.symbol_usage_map[item] -= 1
        self.symbol_usage_map[symbol] += 1

    def _execute_eviction_policy(self, symbol):
        # Type: Least Recently Used
        self.LRU_symbol = min(self.symbol_usage_map, key= self.symbol_usage_map.get)
        #print(self.LRU_symbol)
        #print(np.where(self.l1 == self.LRU_symbol))
        self.LRU_symbol_index = int(np.where(self.l1 == self.LRU_symbol)[0])
        evicted_symbol = self.l1[self.LRU_symbol_index]

        ram_symbol_index = int(np.where(self.ram == symbol)[0])
        self.ram[ram_symbol_index] = evicted_symbol
        self.l1[self.LRU_symbol_index] = symbol

        self.symbol_usage_map[symbol] = 0
        self.symbol_usage_map[evicted_symbol] = 1


    def _populate_cache(self, permutation):
        symbol_iter = iter(permutation)
        try:
            for i in range(self.tcm_size):
                self.tcm[i] = next(symbol_iter)
            for i in range(self.l1_size):
                self.l1[i] = next(symbol_iter)
            for i in range(int(1e6)):
                self.ram[i] = next(symbol_iter)
        except StopIteration:
            pass
        for item in self.l1:
            self.symbol_usage_map[item] = 0
    
    def _empty_cache(self):
        self.tcm = np.empty(self.tcm_size, dtype=object)
        self.l1 = np.empty(self.l1_size, dtype=object)
        self.ram = np.empty(int(1e6), dtype=object)
        self.LRU_symbol = None
        self.symbol_usage_map = {k:1 for k in self.symbol_list}
        

if __name__ == "__main__":
    symbols = np.array(range(100))
    plog = np.random.choice(symbols, 10000)
    sim = HRTSim(plog, 10, 20, 1, 0.1, 1.5)
    permutation = np.array(range(100))
    np.random.shuffle(permutation)
    print(sim(permutation))