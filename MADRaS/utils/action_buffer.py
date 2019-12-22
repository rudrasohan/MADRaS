from queue import Queue 
from copy import deepcopy
import numpy as np
import MADRaS.utils.madras_datatypes as md

madras = md.MadrasDatatypes()

class ActionBuffer:
    """
        To hold the last k actions of taken 
        by an agent. 
    """

    def __init__(self, agent_id, size, action_dim):
        self._agent_id = agent_id
        self._size = size
        self._action_dim = action_dim
        self._buffer = Queue(maxsize=self._size)
        self._curr_size = 0

    def insert(self, action):
        
        if (self._curr_size < self._size):
            self._curr_size += 1
            self._buffer.put(action)
        else: 
            _ = self._buffer.get()
            self._buffer.put(action)

    def request(self):
        temp_queue = Queue(maxsize=self._size)
        ret = np.zeros((self._size*self._action_dim,), dtype=madras.floatX)
        for i in range(self._curr_size):
            a = self._buffer.get()
            temp_queue.put(a)
            ret[i*self._action_dim: (i+1)*self._action_dim] = a
        self._buffer = temp_queue
        return ret

    def reset(self):
        del self._buffer
        self._buffer = Queue(maxsize=self._size)
        self._curr_size = 0
