import random
import pickle

class DataIO(object):
    def __init__(self, dataset=None, batch_size=64, ts=-1):
        self.idx = 0

        if dataset is not None:
            if type(dataset) is str:
                with open(dataset, 'rb') as fp:
                    self.data = pickle.load(fp)
                random.seed(0)
                random.shuffle(self.data)
                if ts != -1:
                    self.data = self.data[0:ts] # use small sub-dataset for training
                self.batch_size = min(batch_size, len(self.data))
            elif type(dataset) is list:
                self.data = dataset
                self.batch_size = min(batch_size,len(self.data))
            else:
                raise Exception("Unsupported dataset type!")
            
        else:
            raise Exception("Input dataset is None!")
    
    def _next_batch(self):
        """ Get next batch data

        Use `self.batch` to get current batch data, call this function will update the batch data.
        """
        batch = self.data[self.idx:self.idx+self.batch_size]
        self.idx += self.batch_size
        
        if self.idx >= len(self.data):
            self.idx = self.idx % len(self.data)
            batch += self.data[:self.idx]
        
        self.batch = batch
    
    def reset(self, keep=False):
        """ Reset dataIO
        
        Args:
            keep: call `self._next_batch()` or not to update `self.batch`, default is False
        """

        if not keep:
            self._next_batch()

    def split(self, num):
        """ Split batch to sub-batch for multi-process
        Args:
            num: the number of processes
        
        Returns:
            A list contains sub-envs(`DataIO2`).
        """
        sub_batch_size = int(self.batch_size / num)
        sub_dataIOs = []
        n = list([int(len(self.data) * 1.0 * i / num) for i in range(num+1)])
        data_split = []
        
        for i in range(num):
            data_split.append(self.data[n[i]:n[i+1]])
        
        for i in range(num):
            sub_dataIO = DataIO(data_split[i], sub_batch_size)
            sub_dataIOs.append(sub_dataIO)
        
        return sub_dataIOs