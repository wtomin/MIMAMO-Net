import torch
from torch._six import int_classes as _int_classes
class Sampler(object):
    """Base class for all Samplers.

    Every Sampler subclass has to provide an __iter__ method, providing a way
    to iterate over indices of dataset elements, and a __len__ method that
    returns the length of the returned iterators.
    """

    def __init__(self, data_source):
        pass

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
        
class SubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)
    
class SameLengthBatchSampler(Sampler):
    """indices_list is a list, each element is another list, consisting of all indices of same lengths. Each time sample from the same list
       Arguments:
           indices_list: a list of sublists, each sublist is a sequence of indices
    """
    def __init__(self, indices_list, batch_size, drop_last, random=True):
        self.indices_list = indices_list
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.random = random
    def shuffle(self, indices):
        length = len(indices)
        r = torch.randperm(length)
        return [indices[id] for id in r]
    def __iter__(self):
        for indices in self.indices_list:
            batch = []
            for idx in indices:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch if not self.random else self.shuffle(batch)
                    batch = []
            if len(batch) > 0 and not self.drop_last:
                yield batch if not self.random else self.shuffle(batch)
    def __len__(self):
        indices_num = [len(indices) for indices in self.indices_list]
        if self.drop_last:
            return sum(indices_num) // self.batch_size
        else:
            return (sum(indices_num) + self.batch_size - 1) // self.batch_size