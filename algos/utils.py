import itertools
from pprint import pprint
import numpy as np

import torch
import torch.optim as optim
import torch_scatter
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

import deep_logic

from algos.datasets import BFSSingleIterationDataset, ParallelColoringSingleGeneratorDataset, ParallelColoringDataset, CombinedGeneratorsDataset
from algos.hyperparameters import get_hyperparameters
import algos.models as models

def flip_edge_index(edge_index):
    return torch.stack((edge_index[1], edge_index[0]), dim=0)

_DEVICE = get_hyperparameters()['device']
_DIM_LATENT = get_hyperparameters()['dim_latent']

def prepare_batch(batch):
    batch = batch.to(_DEVICE)
    if len(batch.x.shape) == 2:
        batch.x = batch.x.unsqueeze(-1)
    batch.x = batch.x.transpose(1, 0)
    batch.y = batch.y.transpose(1, 0)
    batch.termination = batch.termination.transpose(1, 0)
    batch.num_nodes = len(batch.x[0])
    return batch

def get_graph_embedding(latent_nodes, batch_ids, reduce='mean'):
    graph_embs = torch_scatter.scatter(latent_nodes, batch_ids, dim=0, reduce=reduce)
    return graph_embs

def _test_get_graph_embedding():
    latent_nodes = torch.tensor([[1., 5.], [3., 8.], [15., 20.]])
    batch_ids = torch.tensor([0, 0, 1])
    assert torch.allclose(get_graph_embedding(latent_nodes, batch_ids), torch.tensor([[2., 6.5], [15., 20.]]))

def get_mask_to_process(continue_logits, batch_ids, debug=False):
    """

    Used for graphs with different number of steps needed to be performed

    Returns:
    mask (1d tensor): The mask for which nodes still need to be processed

    """
    if debug:
        print("Getting mask processing")
        print("Continue logits:", continue_logits)
    mask = continue_logits[batch_ids] > 0
    if debug:
        print("Mask:", mask)
    return mask

def _test_get_mask_to_process():
    cp = torch.tensor([0.78, -0.22])
    batch_ids = torch.tensor([0, 0, 1])
    assert (get_mask_to_process(cp, batch_ids, debug=True) == torch.tensor([True, True, False])).all()


def load_algorithms_and_datasets(algorithms,
                                 processor,
                                 dataset_kwargs,
                                 bias=False,
                                 use_TF=False,
                                 L1_loss=False,
                                 prune_logic_epoch=-1,
                                 pooling='attention',
                                 next_step_pool=True,
                                 new_coloring_dataset=False,
                                 get_attention=False,
                                 **kwargs):
    for algorithm in algorithms:
        algo_class = models.AlgorithmBase if algorithm == 'BFS' else models.AlgorithmColoring
        inside_class = BFSSingleIterationDataset if algorithm == 'BFS' else None
        dataclass = CombinedGeneratorsDataset if algorithm == 'BFS' or new_coloring_dataset else ParallelColoringDataset
        rootdir = f'./algos/{algorithm}'

        algo_net = algo_class(
            latent_features=get_hyperparameters()[f'dim_latent'],
            node_features=get_hyperparameters()[f'dim_nodes_{algorithm}'],
            output_features=get_hyperparameters()[f'dim_target_{algorithm}'],
            algo_processor=processor,
            dataset_class=dataclass,
            inside_class=inside_class,
            dataset_root=rootdir,
            dataset_kwargs=dataset_kwargs[algorithm],
            bias=bias,
            use_TF=use_TF,
            L1_loss=L1_loss,
            prune_logic_epoch=prune_logic_epoch,
            global_termination_pool=pooling,
            next_step_pool=next_step_pool,
            get_attention=get_attention
        ).to(_DEVICE)
        processor.add_algorithm(algo_net, algorithm)

def iterate_over(processor,
                 optimizer=None,
                 return_outputs=False,
                 num_classes=2,
                 hardcode_outputs=False,
                 epoch=None,
                 batch_size=None,
                 aggregate=False,**lw):

    done = {}
    iterators = {}
    for name, algorithm in processor.algorithms.items():
        algorithm.epoch = epoch
        iterators[name] = iter(DataLoader(
            algorithm.dataset,
            batch_size=get_hyperparameters()['batch_size'] if batch_size is None else batch_size,
            shuffle=False,
            drop_last=False))
        done[name] = False
        algorithm.zero_validation_stats()

    idx = 0
    while True:
        for name, algorithm in processor.algorithms.items():
            try:
                algorithm.step_idx += 1
                algorithm.zero_steps()
                algorithm.zero_tracking_losses_and_statistics()
                batch = next(iterators[name])
                with torch.set_grad_enabled(processor.training):
                    algorithm.process(batch,
                                      hardcode_outputs=hardcode_outputs)
                    # Wrong flag and wrong indices are collected
                    # if we want to check which batch/sample was misclassified
                    if algorithm.wrong_flag and not algorithm.training:
                        algorithm.wrong_indices.append(idx)

            # All algorithms are iterated for at most |nodes| steps
            except StopIteration:
                done[name] = True
                continue
        if processor.training and not all(done.values()):
            processor.update_weights(optimizer)
        idx += 1
        if all(done.values()):
            break

def toggle_freeze_module(module):
    for param in module.parameters():
        param.requires_grad ^= True

if __name__ == '__main__':
    _test_get_graph_embedding()
    _test_get_mask_to_process()

    bs = True
    algo = ['BFS', 'parallel_coloring']
    processor = models.AlgorithmProcessor(_DIM_LATENT, bias=bs, use_gru=False).to(_DEVICE)
    load_algorithms_and_datasets(algo, processor, {'split': 'train', 'generator': 'ER', 'num_nodes': 20}, bias=bs)
    optimizer = optim.Adam(processor.parameters(),
                           lr=get_hyperparameters()[f'lr'],
                           weight_decay=get_hyperparameters()['weight_decay'])
    pprint(processor.state_dict().keys())
    print(processor)
    for epoch in range(2000):
        processor.train()
        processor.load_split('train')
        iterate_over(processor, optimizer=optimizer, epoch=epoch)
        if (epoch+1) % 1 == 0:
            processor.eval()
            print("EPOCH", epoch)
            for spl in ['val']:
                processor.load_split(spl)
                iterate_over(processor, epoch=epoch)
                for name, algorithm in processor.algorithms.items():
                    print("algo=", name)
                    pprint(algorithm.get_losses_dict(validation=True))
                    pprint(algorithm.get_validation_accuracies())
                exit(0)

    processor.eval()
    processor.load_split('test')
    iterate_over(processor, extract_formulas=True, epoch=0)
    iterate_over(processor, epoch=0)

    for algorithm in processor.algorithms.values():
        print(algorithm.get_validation_accuracies())
