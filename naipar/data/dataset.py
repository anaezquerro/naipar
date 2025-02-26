from __future__ import annotations
from typing import List, Tuple, Optional, Dict 
from torch.utils.data import Dataset
import random
        
    
class AbstractDataset(Dataset):
    SEP = None
    EXTENSION = None
    END = ''
    HEADER = ''
    
    def __init__(self, sens: List, name: str):
        self.sens = sens 
        self.name = name 
        for i, sen in enumerate(self.sens):
            if sen.ID is None:
                sen.ID = i 
        self.sort()
        self._distro = None 
                
    def sort(self) -> AbstractDataset:
        self.sens = sorted(self.sens, key=lambda sen: sen.ID)
        return self 
        
    def __iter__(self):
        return iter(self.sens)
    
    def __getitem__(self, index: int):
        if isinstance(index, int):
            return self.sens[index]
        else:
            return self.__class__(self.sens[index], self.name)
    
    def __len__(self) -> int:
        return len(self.sens)
    
    def split(self, p: float, shuffle: bool = True) -> Tuple[AbstractDataset, AbstractDataset]:
        if shuffle:
            random.shuffle(self.sens)
        n = int(p*len(self))
        split = self.sens[:n]
        self.sens = self.sens[n:]
        return self, self.__class__(split, self.name)
        
    def copy(self) -> AbstractDataset:
        return self.__class__([sen.copy() for sen in self.sens], name)
    
    def join(self, *others: List[AbstractDataset], name: Optional[str] = None) -> AbstractDataset:
        sens = self.sens 
        for other in others:
            sens += other.sens 
        return self.__class__(sens, name or self.name)
    
    def save(self, path: str):
        self.sort()
        with open(path, 'w') as writer:
            if len(self.HEADER) > 0:
                writer.write(self.HEADER + '\n')
            writer.write(self.SEP.join(sen.format() for sen in self.sens))
            writer.write(self.END)
            
    @classmethod
    def from_files(cls, paths: List[str], name: Optional[str] = None, num_workers: int = 1) -> AbstractDataset:
        datas = [cls.from_file(path, num_workers=num_workers) for path in paths]
        data = datas.pop(0)
        data.join(*datas, name=name)
        return data

    @property
    def n_tokens(self) -> int:
        return sum(map(len, self.sens))
    
    @property
    def lens(self) -> List[int]:
        return list(map(len, self.sens))
    
    @property
    def distro(self):
        raise NotImplementedError
    
    def pop(self, start: int, end: int) -> AbstractDataset:
        partial = self.sens[start:end]
        self.sens = self.sens[:-start] + self.sens[end:]
        return self.__class__(partial, self.name)