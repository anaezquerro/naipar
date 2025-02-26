from argparse import ArgumentParser
import os, torch 

from naipar import ZeroShotDependencyParser
from naipar.utils import parallel
from naipar.utils.metric import DependencyMetric
from naipar.data import CoNLL


os.environ["TOKENIZERS_PARALLELISM"] = "false"
PRECISION = {32: torch.float32, 16: torch.bfloat16}

BASELINES = {
    'left': CoNLL.Graph.left_branch,
    'right': CoNLL.Graph.right_branch,
    'random': CoNLL.Graph.random,
    'random-proj': CoNLL.Graph.random_projective,
    'linear': CoNLL.Graph.optimal_linear,
    'linear-proj': CoNLL.Graph.optimal_linear_projective,
}

if __name__ == '__main__':
    argparser = ArgumentParser(description='Zero-shot Parsing')
    argparser.add_argument('type', type=str, choices=['zero', 'left', 'right', 'random', 'random-proj', 'linear', 'linear-proj', 'sample'], help='Select the type of approach')
    argparser.add_argument('--pretrained', type=str, default=None, help='Select the LLM from HuggingFace repository')
    argparser.add_argument('--path', type=str, help='Path to store results')
    argparser.add_argument('--data', type=str, help='CoNLL file to perform prediction.')
    argparser.add_argument('--ref', type=str, help='Reference treebank to extract somee statistics')
    argparser.add_argument('--batch-size', type=int, default=50, help='Batch size')
    argparser.add_argument('--precision', type=int, choices=PRECISION.keys(), default=32, help='Precision')
    argparser.add_argument('--load4bit', action='store_true', help='Load in 4 bits')
    argparser.add_argument('--temperature', type=float, default=1.0, help='Temperature value for generation')
    argparser.add_argument('--num-workers', type=int, default=1, help='Number of workers to parallelize execution')

    args = argparser.parse_args()
    
    data = CoNLL.from_file(args.data, num_workers=args.num_workers)
    ref = CoNLL.from_file(args.ref, num_workers=args.num_workers)
    args.REL = ref.REL
    os.makedirs(args.path, exist_ok=True)
    
    if args.type == 'zero':
        parser = ZeroShotDependencyParser(args.pretrained, dtype=PRECISION[args.precision], load_in_4bit=args.load4bit, temperature=args.temperature)
        post1, post2 = parser.evaluate(ref, data, args.path, batch_size=args.batch_size)
        print(f'Final metric after the first postprocessing step: {post1}')
        print(f'Final metric after the second postprocessing step: {post2}')
        post1.save(f'{args.path}/post1.pickle')
        post2.save(f'{args.path}/post2.pickle')
    else:
        if args.type == 'sample':
            pred = parallel(ref.sample, data, [args.REL for _ in data], num_workers=args.num_workers, name=args.type)
        else:
            pred = parallel(BASELINES[args.type], data, [args.REL for _ in data], num_workers=args.num_workers, name=args.type)
        pred = CoNLL(pred, name=None)
        pred.save(f'{args.path}/{data.name}.conllu')
        metric = DependencyMetric(data, pred)
        metric.save(f'{args.path}/{data.name}.pickle')
        print(f'Final metric: {metric}')
    
    
