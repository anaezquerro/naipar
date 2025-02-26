from typing import Iterator, Callable, List 
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm 
import numpy as np 
import os, torch, random, shutil 

def div(num: float, den: float) -> float:
    if den == 0:
        return 0 
    else:
        return num/den


def avg(items) -> float:
    total, n = 0, 0
    for item in items:
        total += item 
        n += 1
    return total/n

def listdir(folder: str, absolute: bool = False) -> List[str]:
    files = os.listdir(folder)
    if absolute:
        files = [f'{folder}/{file}' for file in files]
    return files 

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.random.manual_seed(seed)

def flatten(*lists, levels: int = -1) -> list:
    result = []
    for item in lists:
        if (isinstance(item, list) or isinstance(item, tuple) or isinstance(item, Iterator) or isinstance(item, set)) \
            and levels != 0:
            result += flatten(*item, levels=levels-1)
        else:
            result.append(item)
    return result 

def mmap(func: Callable, name: str, *data):
    data = list(zip(*data))
    return [func(*x) for x in tqdm(data, total=len(data), desc=name, leave=False)]
        
def parallel(func: Callable, *data, num_workers: int = os.cpu_count(), name: str = ''):
    if num_workers <= 1:
        return mmap(func, name, *data)
    results = []
    min_len = min(map(len, data))
    batch_size = int(min_len//num_workers+0.5)
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = []
        for i in range(0, min_len, batch_size):
            partial = [x[i:(i+batch_size)] for x in data]
            futures.append(pool.submit(mmap, func, name, *partial))
        for f in futures:
            results += f.result()
    return results 

def pad(tensors: List[torch.Tensor], value: int = 0, side: str = 'right') -> torch.Tensor:
    max_len = max(x.shape[-1] for x in tensors)
    if side == 'right':
        _pad = lambda x: torch.cat([x, torch.full((*x.shape[:-1], max_len-x.shape[-1]), value).to(x.device)])
    else:
        _pad = lambda x: torch.cat([torch.full((*x.shape[:-1], max_len-x.shape[-1]), value).to(x.device), x])
    return torch.stack([_pad(x) for x in tensors])
    
def init_folder(path: str):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    os.makedirs(path)
    
def remove(path: str):
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

def div(num: float, den: float) -> float:
    num, den = map(float, (num, den))
    if den == 0:
        return 0 
    else:
        return num/den