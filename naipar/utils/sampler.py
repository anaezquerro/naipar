from torch.utils.data import Sampler 
import random 
            
class StrictTokenizationSampler(Sampler):
    def __init__(self, data, batch_size: int, shuffle: bool = False):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = data 
        self.step()
        buckets = {}
        for i, sen in enumerate(data):
            try:
                buckets[len(sen)].append(i)
            except KeyError:
                buckets[len(sen)] = [i]
        self.buckets = list(buckets.values())
        
    def __iter__(self):
        batch, total = [], 0
        for bucket in self.buckets:
            for i in bucket:
                length = len(self.data[i])
                if ((total + length) > self.batch_size and len(batch) > 0):
                    yield batch 
                    batch, total = [], 0
                batch.append(i)
                total += length 
        yield batch
        
    def step(self):
        if self.shuffle:
            for bucket in self.buckets:
                random.shuffle(bucket)
            random.shuffle(self.buckets)
            
    def __len__(self) -> int:
        return len(list(iter(self)))