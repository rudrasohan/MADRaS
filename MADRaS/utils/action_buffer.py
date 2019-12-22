from queue import Queue 
from copy import copy
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
            self._buffer.put(action)
            self._curr_size += 1
        else: 
            self._buffer.get()
            self._buffer.put(action)

    def request(self):
        temp_queue = copy(self._buffer)
        ret = np.zeros((self._size*self._action_dim,), dtype=madras.floatX)
        for i in range(self._curr_size):
            ret[i: i+self._action_dim] = temp_queue.get()
        del temp_queue
        return ret

    def reset(self):
        del self._buffer
        self._buffer = Queue(maxsize=self._size)
        self._curr_size = 0
