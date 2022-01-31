"""
Script to test a specific serialised model (.pt PyTorch file).

Usage:
    test.py (--model-path MP) [--algos=ALGO]... [options] 

Options:
    -h --help              Show this screen.

    --algos ALGO           Which algorithms to load {BFS, parallel_coloring}.
                           Repeatable parameter. [default: BFS]

    --has-GRU              Does the processor utilise GRU unit? [default: False]

    --pooling=PL           What graph pooling mechanism to use for termination.
                           One of {attention, predinet, max, mean}. [default: predinet]

    --no-next-step-pool    Do NOT use next step information for termination.
                           Use current instead. [default: False]

    --model-path MP        Path of the model to load

    --num-nodes NN         Number of nodes in the graphs to test on [default: 20]

    --all-num-nodes        Just do all 20, 50, 100 nodes' tests. [default: False]

    --use-seeds            Use seeds for STD. It will automatically modify name as
                           appropriate. [default: False]

    --num-seeds NS         How many different seeds to plot. [default: 5]

"""
import torch
import torch_geometric
from torch_geometric.data import DataLoader
import networkx as nx
import matplotlib.pyplot as plt
import schema
from pprint import pprint
from docopt import docopt
from deep_logic.utils.layer import prune_logic_layers
from algos.models import AlgorithmProcessor
from algos.hyperparameters import get_hyperparameters
from algos.utils import iterate_over, load_algorithms_and_datasets
import seaborn as sns

hardcode_outputs = False
def test_model(processor, path, num_nodes):
    print("PATH", path)
    processor.load_state_dict(torch.load(path))
    processor.eval()
    _BATCH_SIZE = get_hyperparameters()['batch_size']

    processor.load_split('test', num_nodes=num_nodes)

    iterate_over(processor, epoch=0, batch_size=_BATCH_SIZE, hardcode_outputs=hardcode_outputs)
    print("Accuracy")
    accs = {}
    for name, algorithm in processor.algorithms.items():
        pprint(f"{name} ACC")
        pprint(algorithm.get_losses_dict(validation=True))
        accs[name] = algorithm.get_validation_accuracies()
        pprint(algorithm.get_validation_accuracies())

    return accs

torch.cuda.manual_seed(0)
args = docopt(__doc__)
schema = schema.Schema({'--algos': schema.And(list, [lambda n: n in ['BFS', 'parallel_coloring']]),
                        '--help': bool,
                        '--has-GRU': bool,
                        '--pooling': schema.And(str, lambda s: s in ['attention', 'predinet', 'mean', 'max']),
                        '--no-next-step-pool': bool,
                        '--use-seeds': bool,
                        '--num-seeds': schema.Use(int),
                        '--num-nodes': schema.Use(int),
                        '--all-num-nodes': schema.Use(int),
                        '--model-path': schema.Or(None, schema.Use(str))})
args = schema.validate(args)
print(args)

_DEVICE = get_hyperparameters()['device']
_DIM_LATENT = get_hyperparameters()['dim_latent']
processor = AlgorithmProcessor(
    _DIM_LATENT,
    bias=get_hyperparameters()['bias'],
    use_gru=args['--has-GRU'],
).to(_DEVICE)

load_algorithms_and_datasets(args['--algos'],
                             processor,
                             {
                                 'BFS': {
                                     'split': 'train',
                                     'generators': get_hyperparameters()['generators_BFS'],
                                     'num_nodes': 20,
                                     # 'datapoints_train': 10,
                                     # 'datapoints_non_train': 1,
                                 },
                                 'parallel_coloring': {
                                     'split': 'train',
                                     'num_nodes': 20,
                                     # 'datapoints_train': 10,
                                     # 'datapoints_non_train': 1,
                                 }
                             },
                             use_TF=False, # not used when testing
                             get_attention=True and args['--pooling'],
                             pooling=args['--pooling'],
                             next_step_pool=not args['--no-next-step-pool'],
                             bias=get_hyperparameters()['bias'])

if not args['--use-seeds']:
    processor.load_state_dict(torch.load(args['--model-path']))
    test_model(processor, args['--model-path'], args['--num-nodes'])

else:

    def test_num_nodes(num_nodes):
        targets = get_hyperparameters()[f'dim_target_{args["--algos"][0]}']
        per_seed_accs = []
        for seed in range(args['--num-seeds']):
            mp = f'{args["--model-path"]}_seed_{seed}.pt'
            accs = test_model(processor, mp, num_nodes)
            per_seed_accs.append(accs)

        def combine_seed(LD, algo_names):
            DL = {}
            for name in algo_names:
                apsa = {k: [dic[name][k] for dic in LD] for k in LD[0][name].keys()}
                for key in list(apsa.keys()):
                    if not torch.is_tensor(apsa[key]):
                        apsa[key] = [torch.tensor(ak).float().to(_DEVICE) for ak in apsa[key]]
                    apsa[key] = torch.stack(apsa[key], dim=0)
                    apsa[key+'_mean'] = torch.mean(apsa[key], dim=0)
                    apsa[key+'_std'] = torch.std(apsa[key], dim=0)
                DL[name] = apsa
            return DL

        algo_per_seed_accs = combine_seed(per_seed_accs, processor.algorithms.keys())
        print("num nodes", num_nodes)
        pprint(algo_per_seed_accs)

        return algo_per_seed_accs

    algo_per_seed_accs = test_num_nodes(args['--num-nodes'])
    if args['--all-num-nodes']:
        algo_per_seed_accs_per_numnodes = {}
        for nn in [20, 50, 100]:
            algo_per_seed_accs = test_num_nodes(nn)
            algo_per_seed_accs_per_numnodes[nn] = algo_per_seed_accs

        print("ALL NODES")
        pprint(algo_per_seed_accs_per_numnodes)
        print('latex code\n')

        def print_metric(metric, dict_name, dic):
            print(
            f'& {metric} & '
            f'${round(dic[20][args["--algos"][0]][f"{dict_name}"+"_mean"].item()*100, 2)}'
            f'{{\scriptstyle\pm {round(dic[20][args["--algos"][0]][f"{dict_name}"+"_std"].item()*100, 2)}\%}}$ '
            f'& ${round(dic[50][args["--algos"][0]][f"{dict_name}"+"_mean"].item()*100, 2)}'
            f'{{\scriptstyle\pm {round(dic[50][args["--algos"][0]][f"{dict_name}"+"_std"].item()*100, 2)}\%}}$ '
            f'& ${round(dic[100][args["--algos"][0]][f"{dict_name}"+"_mean"].item()*100, 2)}'
            f'{{\scriptstyle\pm {round(dic[100][args["--algos"][0]][f"{dict_name}"+"_std"].item()*100, 2)}\%}}$ \\\\')

        print_metric('mean-step acc', 'mean_step_acc', algo_per_seed_accs_per_numnodes)
        print_metric('last-step acc', 'last_step_acc', algo_per_seed_accs_per_numnodes)

        print_metric('term. acc', 'term_mean_step_acc', algo_per_seed_accs_per_numnodes)
