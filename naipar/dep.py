from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch, os, shutil, random, re
from torch.utils.data import DataLoader
from typing import List, Tuple
from tqdm import tqdm 

from naipar.data import CoNLL, Arc
from naipar.data.conll import remove_cycles
from naipar.utils.metric import DependencyMetric
from naipar.utils import StrictTokenizationSampler, parallel, pad, flatten


class ZeroShotDependencyParser:
    def __init__(
        self, 
        pretrained: str,
        dtype: torch.dtype = torch.float32,
        load_in_4bit: bool = False,
        temperature: float = 1.0
    ):
        self.pretrained = pretrained 
        self.tkz = AutoTokenizer.from_pretrained(self.pretrained, padding_side='left')
        self.dtype = dtype 
        self.load_in_4bit = load_in_4bit
        self.max_len = AutoConfig.from_pretrained(pretrained).max_position_embeddings
        self.temperature = temperature
        self.sample = None 
        if self.tkz.pad_token_id is None:
            self.tkz.pad_token = self.tkz.eos_token
        self.pad_index = self.tkz.pad_token_id
        self.special_indexes = list(set(self.tkz.convert_tokens_to_ids(token) for token in flatten(*self.tkz.special_tokens_map.values())))
        self._model = None 
        
    @property
    def model(self) -> AutoModelForCausalLM:
        if self._model is None:
            self._model =  AutoModelForCausalLM.from_pretrained(self.pretrained, torch_dtype=self.dtype, device_map='auto', load_in_4bit=self.load_in_4bit).requires_grad_(False)
        return self._model
        
    def prompt(self, new: CoNLL.Graph) -> str:
        return f'The CoNLL format for the sentence <{self.sample.sentence}> is:\n{self.sample.prompt()}\n.\
            Now return the CoNLL format for the sentence <{new.sentence}>:\n'
        
    def transform(self, graph: CoNLL.Graph):
        if not graph._transformed:
            inputs = self.tkz(self.prompt(graph), return_tensors='pt')
            graph.PROMPT = inputs.input_ids.flatten()
            graph.MASK = inputs.attention_mask.flatten()
            graph._transformed = True 
            
    def filter(self, graph: CoNLL.Graph) -> bool:
        return graph.PROMPT.shape[-1] <= self.max_len
            
    def collate(self, batch: List[CoNLL.Graph]) -> Tuple[torch.Tensor, torch.Tensor, List[CoNLL.Graph]]:
        """Batch collation.
        
        Args:
            batch (List[CoNLL.Graph]): Input graphs.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input indices and attention masks.
        """
        inputs = pad([graph.PROMPT for graph in batch], self.pad_index, side='left').to('cuda')
        mask = pad([graph.MASK for graph in batch], 0, side='left').to('cuda')
        return inputs, mask, batch
        
    @torch.no_grad()
    def eval_step(self, inputs: torch.Tensor, mask: torch.Tensor, graphs: List[CoNLL.Graph]) -> List[str]:
        max_len = max(map(len, graphs))
        outputs = self.model.generate(
            inputs, attention_mask=mask,
            max_new_tokens=11*2*max_len,
            min_new_tokens=10*2*max_len,
            do_sample=True,
            suppress_tokens=self.special_indexes,
            pad_token_id=self.pad_index,
            temperature=self.temperature
        )[:, inputs.shape[-1]:]
        outputs = self.tkz.batch_decode(outputs)
        return outputs
    
    @classmethod
    def clean(cls, output: str, graph: CoNLL.Graph) -> CoNLL.Graph:
        lines = output.split('\n')
        arcs = {}
        for line in lines:
            try:
                values = re.split(r'\s+', line)
                if len(values) > 0 and values[0].isdigit() and int(values[0]) in range(1, len(graph)+1):
                    ID, HEAD, *_ = [int(v) for v in values if v.isdigit()]
                    _, DEPREL, *_ = [v for v in values if not v.isdigit() and v != '_']
                    arcs[ID] = Arc(HEAD, ID, DEPREL)
            except:
                continue 
        if len(arcs) < len(graph):
            no_assigned = set(range(1, len(graph)+1)) - arcs.keys()
            for dep in no_assigned:
                root = random.choice([h for h in range(len(graph)+1) if h != dep])
                arcs[dep] = Arc(root, dep, 'punct')
        return graph.rebuild(sorted(arcs.values()), unique_root=False, acyclic=False)
    
    
    @classmethod 
    def postprocess(cls, output: str, graph: CoNLL.Graph) -> CoNLL.Graph:
        lines = output.split('\n')
        arcs = {}
        for line in lines:
            try:
                values = re.split(r'\s+', line)
                if len(values) > 0 and values[0].isdigit() and int(values[0]) in range(1, len(graph)+1):
                    ID, HEAD, *_ = [int(v) for v in values if v.isdigit()]
                    _, DEPREL, *_ = [v for v in values if not v.isdigit() and v != '_']
                    arcs[ID] = Arc(HEAD, ID, DEPREL if HEAD != 0 else 'root')
            except:
                continue 
        roots = [arc.DEP for arc in arcs.values() if arc.HEAD == 0]
        if len(roots) == 0 and len(arcs) > 0: # no root assigned and there are nodes assigned
            root = random.choice(list(arcs.keys())) # select in assigned 
            arcs[root].HEAD = 0
        elif len(roots) == 0 and len(arcs) == 0: # no root assigned and there are no nodes assigned 
            root = random.choice(list(range(1, len(graph)+1)))
            arcs[root] = Arc(0, root, 'root')
        elif len(roots) > 1: # more than one root assigned 
            root = random.choice(roots)
        else:
            root = roots.pop(0)
        for arc in arcs.values():
            if arc.DEP != root and (arc.HEAD not in range(1, len(graph)+1) or arc.HEAD == 0):
                arc.HEAD = root
        if len(arcs) < len(graph):
            no_assigned = set(range(1, len(graph)+1)) - arcs.keys()
            for dep in no_assigned:
                arcs[dep] = Arc(root, dep, 'punct')
        return remove_cycles(graph.rebuild(sorted(arcs.values()), unique_root=True, acyclic=False))
    
    @classmethod
    def steps(cls, raw: str, graph: CoNLL.Graph) -> int:
        raw = '\n'.join(line for line in raw.split('\n') if line.split()[0].isdigit())
        try:
            assert len(CoNLL.Graph.from_raw(raw)) == len(graph)
            return 0 
        except:
            try:
                graph.rebuild(cls.clean(raw, graph).arcs)
                return 1 
            except:
                return 2

    def evaluate(
        self,
        train: CoNLL,
        test: CoNLL,
        path: str,
        batch_size: int = 100,
        length: Tuple[int, int] = (4, 7),
        num_workers: int = 1
    ) -> Tuple[DependencyMetric, DependencyMetric]:
        """Zero-shot dependency parsing evaluation.

        Args:
            train (CoNLL): Input CoNLL train dataset.
            test (CoNLL): Input CoNLL test dataset.
            path (str): Path to store results.
        """
        if os.path.exists(path) and not os.path.isdir(path):
            os.remove(path)
        os.makedirs(path, exist_ok=True)
        os.makedirs(f'{path}/log-output', exist_ok=True)
        
        # select random sample for the prompt 
        self.sample = random.choice(list(filter(lambda graph: len(graph) in range(*length), train.sens)))
        parallel(self.transform, test, num_workers=num_workers, name='transform')
        test.sens = list(filter(self.filter, test))
        
        # filter those sentences that are already stored in log-output
        finished = [int(file.split('.')[0]) for file in os.listdir(f'{path}/log-output')]
        remain = CoNLL([sen for sen in test if sen.ID not in finished], test.name)
        print(f'Evaluating {len(remain)} sentences')
        
        sampler = StrictTokenizationSampler(remain, batch_size=batch_size, shuffle=False)
        loader = DataLoader(remain, batch_sampler=sampler, collate_fn=self.collate)
        
        with tqdm(desc='eval', total=len(remain)) as bar:
            for inputs, mask, graphs in loader:
                raws = self.eval_step(inputs, mask, graphs)
                for graph, raw in zip(graphs, raws):
                    with open(f'{path}/log-output/{graph.ID}.txt', 'w') as writer:
                        writer.write(raw)
                bar.update(len(graphs))
        
        # load from log-output
        raws = [open(f'{path}/log-output/{i}.txt', 'r').read() for i in range(len(test))]
        post1 = CoNLL(list(map(self.clean, raws, test)), test.name)
        post2 = CoNLL(list(map(self.postprocess, raws, test)), test.name)
        test.save(f'{path}/gold.conllu')
        post1.save(f'{path}/post1.conllu')
        post2.save(f'{path}/post2.conllu')
        return DependencyMetric(post1, test), DependencyMetric(post2, test)


            

        
        
    