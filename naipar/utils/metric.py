from __future__ import annotations
from typing import List, Dict, Union
import pickle 


from naipar.data import CoNLL
from naipar.utils.fn import parallel, div 


class Metric:
    METRICS = [] 
    ATTRIBUTES = []
    KEY_METRICS = []
    
    def __init__(self, *args, **kwargs):
        for attr in self.ATTRIBUTES:
            self.__setattr__(attr, 0.0)
        
        if len(args) > 0 or len(kwargs) > 0:
            self(*args, **kwargs)
    
    def __add__(self, other: Metric) -> Metric:
        if isinstance(other, Metric):
            for attr in self.ATTRIBUTES:
                self.__setattr__(attr, getattr(self, attr) + getattr(other, attr))
        return self
    
    def __radd__(self, other: Metric) -> Metric:
        return self + other if isinstance(other, Metric) else self 
    
    def __call__(self, *args, **kwargs) -> Metric:
        raise NotImplementedError
    
    def __repr__(self):
        return f', '.join(f'{name.upper()}={round(float(getattr(self, name)*100), 2)}' for name in self.METRICS)
    
    def __getitem__(self, name: str) -> float:
        return self.__getattribute__(name)
    
    def improves(self, other: Metric) -> bool:
        assert all(k1 == k2 for k1, k2 in zip(self.KEY_METRICS, other.KEY_METRICS)) 
        return any(getattr(self, k1) > getattr(other, k2) for k1, k2 in zip(self.KEY_METRICS, other.KEY_METRICS))

    def save(self, path: str):
        with open(path, 'wb') as writer:
            pickle.dump(self, writer)
            
    def values(self, scale: float = 1) -> List[float]:
        return [getattr(self, name)*scale for name in self.METRICS]

    def items(self) -> Dict[str, float]:
        return {name: getattr(self, name) for name in self.METRICS}
    
    @classmethod
    def load(cls, path: str) -> Metric:
        with open(path, 'rb') as reader:
            metric = pickle.load(reader)
        return metric 


class DependencyMetric(Metric):
    ATTRIBUTES = ['uas', 'las', 'ucm', 'lcm', 'n']
    METRICS = ['UAS', 'LAS', 'UCM', 'LCM']
    
    def __call__(
        self, 
        pred: Union[CoNLL, CoNLL.Graph], 
        gold: Union[CoNLL, CoNLL.Graph], 
        num_workers: int = 1
    ) -> DependencyMetric:
        if isinstance(pred, CoNLL.Graph) and isinstance(gold, CoNLL.Graph):
            self.apply(pred, gold)
        else:
            self += sum(parallel(DependencyMetric, pred, gold, num_workers=num_workers, name='dep-metric'))
        return self 
        
    def apply(self, pred: CoNLL.Graph, gold: CoNLL.Graph) -> DependencyMetric:
        assert len(pred.heads) == len(gold.heads), f'Number of heads must match:\n{pred.format()}\n{pred.heads}\n{gold.format()}\n{gold.heads}'
        pred_heads, gold_heads = pred.heads, gold.heads 
        pred_rels, gold_rels = pred.rels, gold.rels 
        
        umask = (pred_heads == gold_heads)
        lmask = (pred_heads == gold_heads) & (pred_rels == gold_rels)
        self.uas += umask.mean()
        self.las += lmask.mean() 
        self.ucm += umask.all()
        self.lcm += lmask.all()
        self.n += 1
        return self 
    
    @property
    def UAS(self) -> float:
        return div(self.uas, self.n)
    
    @property
    def LAS(self) -> float:
        return div(self.las, self.n)
    
    @property
    def UCM(self) -> float:
        return div(self.ucm, self.n)
    
    @property
    def LCM(self) -> float:
        return div(self.lcm, self.n)
    
