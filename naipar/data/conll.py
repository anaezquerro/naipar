from __future__ import annotations
from typing import List, Set, Optional, Union, Tuple, Dict 
import numpy as np 
import random

from naipar.data.arc import Arc 
from naipar.data.dataset import AbstractDataset


class CoNLL(AbstractDataset):
    """Representation of the CoNLL-U format (https://universaldependencies.org/format.html)"""

    HEADER = ''
    EXTENSION = 'conllu'
    SEP = '\n\n'
    END = '\n\n'

    class Node:
        """Abstract implementation of a node in a Dependency Graph."""
        FIELDS = ['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']
        SEP = '\t'
        BOS = '<bos>'
        EOS = '<eos>'
        
        def __init__(
            self, 
            ID: Union[int, str], 
            FORM: str, 
            LEMMA: str, 
            UPOS: str, 
            XPOS: str, 
            FEATS: str, 
            HEAD: Optional[Union[str, int]], 
            DEPREL: str, 
            DEPS: str, 
            MISC: str
        ):
            self.ID = int(ID)
            self.FORM = FORM 
            self.LEMMA = LEMMA 
            self.UPOS = UPOS 
            self.XPOS = XPOS 
            self.FEATS = FEATS
            self.HEAD = int(HEAD) if HEAD is not None else None 
            self.DEPREL = DEPREL
            self.DEPS = DEPS 
            self.MISC = MISC 
        
        def __repr__(self) -> str:
            return self.format()
        
        def __eq__(self, other: CoNLL.Node) -> bool:
            return isinstance(other, self.__class__) and \
                all(v1 == v2 for v1, v2 in zip(self.values(), other.values()))
        
        def __le__(self, other: CoNLL.Node) -> bool:
            return self.ID <= other.ID 
        
        def __lt__(self, other: CoNLL.Node) -> bool:
            return self.ID < other.ID 
        
        def __ge__(self, other: CoNLL.Node) -> bool:
            return self.ID >= other.ID 
        
        def __gt__(self, other: CoNLL.Node) -> bool:
            return self.ID > other.ID
        
        def is_bos(self) -> bool:
            return self.ID == 0
        
        def values(self) -> List[Union[str, int]]:
            return [self.__getattribute__(field) for field in self.FIELDS]
        
        def format(self) -> str:
            assert isinstance(self.HEAD, int), 'NULL HEAD cannot be formatted'
            return self.SEP.join(map(str, [self.__getattribute__(field) for field in self.FIELDS]))
        
        def copy(self) -> CoNLL.Node:
            return CoNLL.Node(*[self.__getattribute__(field) for field in self.FIELDS])
        
        def prompt(self, fields: List[str] = ['ID', 'FORM', 'HEAD', 'DEPREL']) -> str:
            """CoNLL representation for prompting. Only the specified fields are represented, the 
            others are fixed to _ symbol.

            Args:
                fields (List[str], optional): Represented fields. Defaults to ['ID', 'FORM', 'HEAD', 'DEPREL'].

            Returns:
                str: CoNLL representation for prompting, where only some fields are represented.
            """
            return self.SEP.join(str(self.__getattribute__(field)) if field in fields else '_' for field in self.FIELDS)
        
        @classmethod
        def bos(cls) -> CoNLL.Node:
            """Create a BoS node with ID set to 0."""
            return cls(0, cls.BOS, cls.BOS, cls.BOS, cls.BOS, cls.BOS, None, cls.BOS, cls.BOS, cls.BOS)

        @classmethod
        def eos(cls, n: int) -> CoNLL.Node:
            """Creates an EoS node with ID set to `n`."""
            return cls(n, cls.EOS, cls.EOS, cls.EOS, cls.EOS, cls.EOS, None, cls.EOS, cls.EOS, cls.EOS)
            
        @classmethod
        def from_raw(cls, line: str) -> CoNLL.Node:
            """Creates a node from CoNLL raw line.

            Args:
                line (str): Input line.

            Returns:
                CoNLL.Node: Built CoNLL node instance.
                
            Examples:
            >>> node = CoNLL.Node.from_raw('1	Al	Al	PROPN	NNP	Number=Sing	0	root	0:root	SpaceAfter=No')
            >>> node.ID, node.FORM
            (1, 'Al')
            """
            values = line.split(cls.SEP)
            if len(values) <= len(CoNLL.Node.FIELDS):
                values += ['_' for _ in range(len(CoNLL.Node.FIELDS) - len(values))]
            if values[6] == '_':
                values[6] = 0
            return CoNLL.Node(*values)
        
    class Graph:
        """Abstract implementation of a Dependency Graph."""
        SEP = '\n'
        
        def __init__(
            self, 
            nodes: List[CoNLL.Node], 
            arcs: List[Arc], 
            annotations: List[str] = [], 
            ID: Optional[int] = None,
            unique_root: bool = True,
            acyclic: bool = True
        ):
            """Initialization of a CoNLL graph.

            Args:
                nodes (List[AbstractNode]): List of nodes.
                arcs (List[Arc]): List of arcs.
                annotations (List[str], optional): Previous list of annotations.
                ID (Optional[int], optional): Integer identifier for sorting. Defaults to None (no sorting).
            """
            # one-headed, acyclic and unique root restrictions 
            assert len(nodes) == len(arcs), f'{nodes}\n{arcs}'
            assert not unique_root or sum(arc.HEAD == 0 for arc in arcs) == 1, 'Graph has more than one root'
            assert not acyclic or not has_cycles(arcs, len(nodes)), f'Graph has cycles\n' + '\n'.join(node.format() for node in nodes)
            self.NODE = CoNLL.Node 
            
            self.nodes = nodes 
            self.arcs = sorted(arcs)
            self.annotations = annotations
            self.ID = ID 
            
            # initialize inner attributes 
            self._planes = None 
            self._transformed = False
            self.build()
            
        def build(self):
            """Add new attributes to the graph corresponding to the node fields."""
            for field in self.NODE.FIELDS:
                if field == 'ID':
                    continue 
                self.__setattr__(field, [getattr(node, field) for node in self.nodes])
                
        def __len__(self) -> int:
            return len(self.nodes)
        
        def __lt__(self, other: CoNLL.Graph) -> bool:
            if isinstance(other, CoNLL.Graph):
                return self.ID < other.ID 
            else:
                raise NotImplementedError
            
        def __eq__(self, other: CoNLL.Graph):
            return (len(other.nodes) == len(self.nodes)) and (len(other.arcs) == len(self.arcs)) and \
                all(node1 == node2 for node1, node2 in zip(sorted(self.nodes), sorted(other.nodes))) and \
                all(arc1 == arc2 for arc1, arc2 in zip(sorted(self.arcs), sorted(other.arcs)))
                
        def __getitem__(self, index: int) -> CoNLL.Node:
            """Returns the node at a given position.
            - Position 0 is reserved for the artificial BOS node.
            - Position i=1..n are reserved for the node i.
            - Position n+1 is reserved for the artificial EOS node.

            Args:
                index (int): Position of the node.

            Returns:
                AbstractNode: _description_
            """
            if index == 0:
                return self.NODE.bos()
            elif index == len(self) + 1:
                return self.NODE.eos(len(self)+1)
            else:
                return self.nodes[index-1]                

        def copy(self) -> CoNLL.Graph:
            return self.__class__([node.copy() for node in self.nodes], [arc.copy() for arc in self.arcs], self.annotations, self.ID)
        
        def format(self, annotations: bool = True) -> str:
            """Formatted representation of the graph."""
            return self.SEP.join((self.annotations if annotations else []) + [node.format() for node in self.nodes])

        def rebuild(self, new_arcs: List[Arc], unique_root: bool = True, acyclic: bool = True) -> CoNLL.Graph:
            nodes = [node.copy() for node in self.nodes]
            # null all nodes
            for node in nodes:
                node.HEAD = None 
                node.DEPREL = None 
            new_arcs = sorted(new_arcs)
            for arc, node in zip(new_arcs, nodes):
                node.HEAD = arc.HEAD 
                node.DEPREL = arc.REL 
            return self.__class__(nodes, new_arcs, self.annotations, self.ID, unique_root, acyclic)
        
        @property
        def planes(self) -> Dict[int, List[Arc]]:
            """Compute the plane distribution in the graph.
            - Arcs to the root node are always assigned to the first plane.
            - The plane assignment is done left-to-right with respect to the nodes of the graph.
            - First planes always are the ones with the largest number of arcs.

            Returns:
                Dict[int, List[Arc]]: Plane distribution.
            """
            if self._planes is None:
                # store the planes 
                self._planes = {0: []}
                for arc in sorted(self.arcs, key=lambda arc: arc.left):
                    added = False
                    for plane in range(len(self._planes)):
                        if not any(arc.cross(other) for other in self._planes[plane]):
                            self._planes[plane].append(arc)
                            added = True 
                            break 
                    if not added:
                        self._planes[len(self._planes)] = [arc]
                # sort and locate in the planes with the most number of arcs 
                order = sorted(self._planes.keys(), key=lambda i: (len(self._planes[i]), i), reverse=True)
                self._planes = {i: self._planes[plane] for i, plane in enumerate(order)}
            return self._planes
        
        @classmethod 
        def from_raw(cls, lines: str, unique_root: bool = True, acyclic: bool = True) -> CoNLL.Graph:
            lines = list(filter(lambda x: len(x) > 0, lines.strip().split('\n')))
            annotations = list(filter(lambda x: x.startswith('#'), lines))
            nodes = list(map(CoNLL.Node.from_raw, filter(lambda x: x.split()[0].isdigit(), lines)))
            arcs = []
            for node in nodes:
                arcs.append(Arc(node.HEAD, node.ID, node.DEPREL))
            return cls(nodes, arcs, annotations, unique_root=unique_root, acyclic=acyclic)
        
        @classmethod
        def left_branch(cls, graph: CoNLL.Graph, REL: str) -> CoNLL.Graph:
            """Create a left-branching dependency tree (the parent of each word is the next one).

            Args:
                graph (CoNLL.Graph): Input graph.
                REL (str): Default arc label.

            Returns:
                CoNLL.Graph: Right-branching tree.
            """
            arcs = [Arc(i+1, i, REL) for i in range(1, len(graph))]
            arcs.append(Arc(0, len(graph), 'root'))
            return graph.rebuild(sorted(arcs))
        
        @classmethod
        def right_branch(cls, graph: CoNLL.Graph, REL: str) -> CoNLL.Graph:
            """Create a right-branching dependency tree (the parent of each word is the previous one).
            For the last word and the root, select a random node.

            Args:
                graph (CoNLL.Graph): Input graph.
                REL (str): Default arc label.

            Returns:
                CoNLL.Graph: Left-branching tree.
            """
            arcs = [Arc(i, i+1, REL) for i in range(len(graph))]
            return graph.rebuild(sorted(arcs))
        
        
        @classmethod
        def random(cls, graph: CoNLL.Graph, REL: str) -> CoNLL.Graph:
            return graph.rebuild(random_dependency_tree(len(graph), REL))
        
        
        @classmethod
        def random_projective(cls, graph: CoNLL.Graph, REL: str) -> CoNLL.Graph:
            arcs = random_dependency_tree(len(graph), REL)
            return graph.rebuild(projectivize(arcs, REL))
        
        @classmethod
        def optimal_linear(cls, graph: CoNLL.Graph, REL: str) -> CoNLL.Graph:
            from lal.linarr import min_sum_edge_lengths, sum_edge_lengths
            from lal.graphs import from_head_vector_to_rooted_tree
            from lal.io import check_correctness_head_vector
            arcs = random_dependency_tree(len(graph), REL)
        
            tree = from_head_vector_to_rooted_tree([arc.HEAD for arc in arcs])
            score, arr = min_sum_edge_lengths(tree)
            
            pos = list(map(int, str(arr).split(' | ')[1][1:-1].split(', ')))
            mapar = {p+1: i+1 for i, p in enumerate(pos)}
            
            new_arcs = []
            for arc in arcs:
                new_arcs.append(Arc(mapar[arc.HEAD] if arc.HEAD != 0 else arc.HEAD, mapar[arc.DEP], REL if arc.HEAD != 0 else 'root'))
            new_arcs = sorted(new_arcs)
            assert len(check_correctness_head_vector([arc.HEAD for arc in new_arcs])) == 0
            new_tree = from_head_vector_to_rooted_tree([arc.HEAD for arc in new_arcs])
            assert sum_edge_lengths(new_tree) == score
            return graph.rebuild(new_arcs)
        
        @classmethod
        def optimal_linear_projective(cls, graph: CoNLL.Graph, REL: str) -> CoNLL.Graph:
            from lal.linarr import min_sum_edge_lengths_projective, sum_edge_lengths
            from lal.graphs import from_head_vector_to_rooted_tree
            from lal.io import check_correctness_head_vector
            arcs = random_dependency_tree(len(graph), REL)
        
            tree = from_head_vector_to_rooted_tree([arc.HEAD for arc in arcs])
            score, arr = min_sum_edge_lengths_projective(tree)
            
            pos = list(map(int, str(arr).split(' | ')[1][1:-1].split(', ')))
            mapar = {p+1: i+1 for i, p in enumerate(pos)}
            
            new_arcs = []
            for arc in arcs:
                new_arcs.append(Arc(mapar[arc.HEAD] if arc.HEAD != 0 else arc.HEAD, mapar[arc.DEP], REL if arc.HEAD != 0 else 'root'))
            new_arcs = sorted(new_arcs)
            assert len(check_correctness_head_vector([arc.HEAD for arc in new_arcs])) == 0
            new_tree = from_head_vector_to_rooted_tree([arc.HEAD for arc in new_arcs])
            assert sum_edge_lengths(new_tree) == score
            return graph.rebuild(new_arcs)
        
        @classmethod
        def base_root(cls, graph: CoNLL.Graph, REL: str) -> CoNLL.Graph:
            root = random.choice(list(range(1, len(graph)+1)))
            arcs = [Arc(0, root, 'root')]
            for dep in range(1, len(graph)+1):
                if dep != root:
                    arcs.append(Arc(root, dep, REL))
            return graph.rebuild(arcs)
        
        @property
        def heads(self) -> np.ndarray:
            return np.array(self.HEAD)
        
        @property
        def rels(self) -> np.ndarray:
            return np.array(self.DEPREL)
        
        @property
        def sentence(self) -> str:
            return ' '.join(self.FORM)
        
        def prompt(self, fields: List[str] = ['ID', 'FORM', 'HEAD', 'DEPREL']) -> str:
            return self.SEP.join(node.prompt(fields) for node in self.nodes)
            

    def __repr__(self) -> str:
        return f'CoNLL(name={self.name}, n={len(self)})'

    @classmethod 
    def from_file(cls, path: str, num_workers: int = 1, name: Optional[str] = None, unique_root: bool = True, acyclic: bool = True) -> CoNLL:
        from naipar.utils import parallel
        blocks = list(filter(lambda x: len(x) > 0, open(path, 'r').read().split('\n\n')))
        graphs = parallel(lambda x: cls.Graph.from_raw(x, unique_root, acyclic), blocks, num_workers=num_workers, name=path.split('/')[-1])
        return cls(graphs, name or path.split('/')[-1].split('.')[0])
    
    @property
    def distro(self) -> Dict[int, Dict[str, int]]:
        if self._distro is None:
            self._distro = {}
            for graph in self.sens:
                arcs = tuple(arc.HEAD for arc in graph.arcs)
                try:
                    self._distro[len(graph)]
                except KeyError:
                    self._distro[len(graph)] = dict()
                try:
                    self._distro[len(graph)][arcs] += 1
                except KeyError:
                    self._distro[len(graph)][arcs] = 1
        return self._distro 
    
    def sample(self, graph: CoNLL.Graph, REL: str) -> CoNLL.Graph:
        """Samples a set of arcs from the dataset given a input set of nodes.

        Args:
            graph (CoNLL.Graph): Input graph (only nodes are taken into account).
            REL (str): Default arc label.

        Returns:
            CoNLL.Graph: Sampled graph.
        """
        try:
            self.distro[len(graph)]
        except KeyError:
            return CoNLL.Graph.random(graph, REL)
        heads, freqs = zip(*self.distro[len(graph)].items())
        freqs = np.array(freqs)
        selected = np.random.choice(range(len(heads)), p=freqs/freqs.sum())
        arcs = [Arc(head, dep+1, 'root' if head == 0 else REL) for dep, head in enumerate(heads[selected])]
        return graph.rebuild(arcs)
                
    @property
    def REL(self) -> str:
        rels, counts = np.unique([arc.REL for graph in self for arc in graph.arcs], return_counts=True)
        return rels[np.argmax(counts)]
    
    @property
    def tags(self) -> Set[str]:
        return set(node.UPOS for graph in self for node in graph.nodes)
 
def random_dependency_tree(n: int, REL: str) -> List[Arc]:
    import lal 
    heads = lal.generate.rand_lab_rooted_trees(n).yield_tree().get_head_vector()
    return [Arc(head, i+1, REL if head != 0 else 'root') for i, head in enumerate(heads)]


def parse_arrangement(arr) -> List[int]:
    new = list(map(lambda x: int(x) + 1, str(arr).split('|')[1].replace('(', '').replace(')', '').split(',')))
    arr = dict(zip(range(1, len(new)+1), new))
    arr[0] = 0
    return arr
    
def projectivize(arcs: List[Arc], REL: str) -> List[Arc]:
    import lal
    rt = lal.graphs.from_head_vector_to_rooted_tree([arc.HEAD for arc in sorted(arcs)])
    arr = lal.generate.rand_projective_arrangements(rt).yield_arrangement()
    arr = parse_arrangement(arr)
    return sorted(Arc(arr[arc.HEAD], arr[arc.DEP], REL if arr[arc.HEAD] != 0 else 'root') for arc in arcs)
    
    

def has_cycles(arcs: List[Arc], n: int) -> bool:
    """Checks whether a list of arcs contains cycles.

    Args:
        arcs (List[Arc]): Arcs of the graph.
        n (int): Number of nodes.

    Returns:
        bool: Whether the set of arcs contain a cycle.
    """
    n_in = [0 for _ in range(n+1)]
    conn = {i: set() for i in range(n+1)}
    queue = []
    
    # update the number of incoming arcs
    for arc in arcs:
        n_in[arc.DEP] += 1
        conn[arc.HEAD].add(arc.DEP)

    # enqueue vertices with 0 in-degree
    for i, value in enumerate(n_in):
        if value == 0:
            queue.append(i)

    visited = 0
    while len(queue) > 0:
        node = queue.pop(0)
        visited += 1
        for neighbor in conn[node]:
            n_in[neighbor] -= 1
            if n_in[neighbor] == 0:
                queue.append(neighbor)

    return visited != (n+1)

        
def propagate(arcs: List[Arc], _plane1: Set[Arc], _plane2: Set[Arc], arc: Arc, i: int) -> Tuple[Set[Arc], Set[Arc]]:
    selected = _plane1 if i == 1 else _plane2
    no_selected = _plane1 if i == 2 else _plane2
    selected.add(arc)
    for other in arcs:
        if arc.cross(other) and other not in no_selected:
            _plane1, _plane2 = propagate(arcs, _plane1, _plane2, other, 3-i)
    return _plane1, _plane2
            
        
def remove_cycles(graph: CoNLL.Graph) -> CoNLL.Graph:
    """
    Removes cycles for a list of arcs where one and only one has the node 0 as head.
    
    Args:
        arcs (List[Arc]): List of input arcs (cyclic and non connected).
    """
    conn = {h: [] for h in range(len(graph)+1)}
    for arc in graph.arcs:
        conn[arc.HEAD].append(arc)
    stack = [0]
    visited = set()
    non_visited = set(range(len(graph)+1))
    arcs = {}
    while len(non_visited) > 0:
        if len(stack) == 0:
            h = non_visited.pop()
        else:
            h = stack.pop(0)
            non_visited.remove(h)
        visited.add(h)
        for arc in conn[h]:
            if arc.DEP not in visited:
                stack.append(arc.DEP)
                arcs[arc.DEP] = arc
            else:
                arcs[arc.DEP] = Arc(conn[0][0].DEP, arc.DEP, arc.REL)
    return graph.rebuild(sorted(arcs.values()))    