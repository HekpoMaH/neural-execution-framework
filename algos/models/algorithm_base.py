import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_scatter
import deep_logic
from deep_logic.utils.layer import l1_loss

import algos.utils as utils
import algos.models as models
from algos.hyperparameters import get_hyperparameters
from algos.layers import GlobalAttentionPlusCoef, PrediNet

_DEVICE = get_hyperparameters()['device']
_DIM_LATENT = get_hyperparameters()['dim_latent']

def printer(module, gradInp, gradOutp):
    print(list(module.named_parameters()))
    s = 0
    mx = -float('inf')
    for gi in gradInp:
        s += torch.sum(gi)
        mx = max(mx, torch.max(torch.abs(gi)))

    print("INP")
    print(f'sum {s}, max {mx}')
    s = 0
    mx = -float('inf')
    print("OUTP")
    for go in gradOutp:
        s += torch.sum(go)
        mx = max(mx, torch.max(torch.abs(go)))
    print(f'sum {s}, max {mx}')
    print(f'gradout {gradOutp}')
    input()

class AlgorithmBase(nn.Module):
    '''
    Base class for algorithm's execution. The class takes into
    account that applying the SAME algorithm to different graphs
    may take DIFFERENT number of steps. The class implementation
    circumvents this problems by masking out graphs (or 'freezing'
    to be precise) which should have stopped executing.

    It also re-calculates losses/metrics, so that they do not differ
    between using batch size of 1 or batch size of >1.
    '''

    @staticmethod
    def get_masks(train, batch, continue_logits, enforced_mask):
        '''
            mask is which nodes out of the batched disconnected graph should
            not change their output/latent state.

            mask_cp is which graphs of the batch should be frozen (mask for
            Continue Probability).

            Once a graph/node is frozen it cannot be unfreezed.

            Masking is important so we don't change testing dynamics --
            testing the same model with batch size of 1 should give the
            same results as testing with batch size of >1.

            continue logits: Logit values for the continue probability
            for each graph in the batch.

            enforced mask: Debugging tool to forcefully freeze a graph.
        '''
        if train:
            mask = utils.get_mask_to_process(continue_logits, batch.batch)
            mask_cp = (continue_logits > 0.0).bool()
            # mask = torch.ones_like(batch.batch).bool()
            # mask_cp = torch.ones_like(continue_logits).bool()
        else:
            mask = utils.get_mask_to_process(continue_logits, batch.batch)
            mask_cp = (continue_logits > 0.0).bool()
            if enforced_mask is not None:
                enforced_mask_ids = enforced_mask[batch.batch]
                mask &= enforced_mask_ids
                mask_cp &= enforced_mask
        return mask, mask_cp

    def load_dataset(self, dataset_class, dataset_root, dataset_kwargs):
        self.dataset_class = dataset_class
        self.dataset_root = dataset_root
        self.dataset_kwargs = dataset_kwargs
        self.dataset = dataset_class(dataset_root, self.inside_class, **dataset_kwargs)

    def __init__(self,
                 latent_features,
                 node_features,
                 output_features,
                 algo_processor,
                 dataset_class,
                 inside_class,
                 dataset_root,
                 dataset_kwargs,
                 bias=False,
                 use_TF=False,
                 L1_loss=False,
                 prune_logic_epoch=-1,
                 global_termination_pool='predinet', #'max',
                 next_step_pool=True,
                 get_attention=False,
                 use_batch_norm=False,
                 **kwargs):

        super(AlgorithmBase, self).__init__()
        self.step_idx = 0
        self.node_features = node_features
        self.output_features = output_features
        self.latent_features = latent_features
        self.debug = False
        self.epoch_threshold_debug = 500
        self.L1_loss = L1_loss
        self.prune_logic_epoch = prune_logic_epoch
        self.global_termination_pool = global_termination_pool
        self.next_step_pool = next_step_pool
        self.processor = algo_processor.processor
        self.use_TF = use_TF
        self.get_attention = get_attention
        self.lambda_mul = 1# 0.0001
        self.inside_class = inside_class
        self.load_dataset(dataset_class, dataset_root, dataset_kwargs)
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features + latent_features, latent_features, bias=bias),
            nn.LeakyReLU()
        )

        self.decoder_network = nn.Sequential(
            nn.Linear(2 * latent_features, latent_features, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(latent_features, output_features, bias=bias)
        )
        # print(dict(self.node_encoder.named_parameters()))
        # print(hash(tuple(torch.get_rng_state().tolist())))
        # print(hash(tuple(torch.cuda.get_rng_state().tolist())))
        # exit(0)

        if global_termination_pool == 'attention':
            inp_dim = latent_features
            self.global_attn = GlobalAttentionPlusCoef(
                    nn.Sequential(
                        nn.Linear(inp_dim, latent_features, bias=bias),
                        nn.LeakyReLU(),
                        nn.Linear(latent_features, 1, bias=bias)
                    ),
                    nn=None)

        if global_termination_pool == 'predinet':
            lf = latent_features
            self.predinet = PrediNet(lf, 1, lf, lf, flatten_pooling=torch_geometric.nn.glob.global_max_pool)

        self.termination_network = nn.Sequential(
            nn.BatchNorm1d(latent_features) if use_batch_norm else nn.Identity(),
            nn.Linear(latent_features, 1, bias=bias),
            )

    def get_continue_logits(self, batch_ids, latent_nodes, sth_else=None):
        if self.global_termination_pool == 'mean':
            graph_latent = torch_geometric.nn.global_mean_pool(latent_nodes, batch_ids)
        if self.global_termination_pool == 'max':
            graph_latent = torch_geometric.nn.global_max_pool(latent_nodes, batch_ids)
        if self.global_termination_pool == 'attention':
            graph_latent, coef = self.global_attn(latent_nodes, batch_ids)
            if self.get_attention:
                self.attentions[self.step_idx] = coef.clone().detach()
                self.per_step_latent[self.step_idx] = sth_else

        if self.global_termination_pool == 'predinet':
            assert not torch.isnan(latent_nodes).any()
            graph_latent = self.predinet(latent_nodes, batch_ids)

        if self.get_attention:
            self.attentions[self.step_idx] = latent_nodes
        continue_logits = self.termination_network(graph_latent).view(-1)
        return continue_logits

    def zero_termination(self):
        self.true_positive = 0
        self.false_positive = 0
        self.false_negative = 0
        self.true_negative = 0

    def zero_steps(self):
        self.sum_of_processed_nodes, self.step_idx, self.sum_of_steps, self.cnt = 0, 0, 0, 0

    def zero_tracking_losses_and_statistics(self):
        if self.training:
            self.zero_termination()
            self.losses = {
                'total_loss_output': 0,
                'total_loss_term': 0,
            }

    def zero_validation_stats(self):
        self.mean_step = []
        self.last_step = []
        self.term_preds_mean_step = []
        self.validation_sum_of_steps = 0
        self.validation_sum_of_processed_nodes = 0
        self.last_step_total = 0
        if self.get_attention:
            self.attentions = {}
            self.per_step_latent = {}
        self.zero_termination()
        self.validation_losses = {
            'total_loss_output': 0,
            'total_loss_term': 0,
        }
        self.losses = {
            'total_loss_output': 0,
            'total_loss_term': 0,
        }
        self.wrong_indices = []

    @staticmethod
    def calculate_step_acc(output, output_real, batch_mask, take_total_for_classes=True):
        """ Calculates the accuracy for a givens step """
        output = output.squeeze(-1)
        correct = 0
        tot = 0
        correct = output == output_real
        if len(correct.shape) == 2 and take_total_for_classes:
            correct = correct.float().mean(dim=-1)
        _, batch_mask = torch.unique(batch_mask, return_inverse=True)
        correct_per_batch = torch_scatter.scatter(correct.float(), batch_mask, reduce='mean', dim=0)
        return correct_per_batch

    def get_output_loss(self, output_logits, target):
        return F.binary_cross_entropy_with_logits(output_logits.squeeze(-1), target, reduction='sum', pos_weight=torch.tensor(1.00))

    def aggregate_step_acc(self, batch, mask, mask_cp, y_curr, output_logits,
                           true_termination, continue_logits):

        masked_batch = batch.batch[mask]
        output_logits_masked = output_logits[mask]
        output_real_masked = y_curr[mask].float()


        assert not torch.isnan(output_logits_masked).any(), output_logits_masked
        assert mask_cp.any()
        # if not mask_cp.all():
        #     print(self.step_idx)
        #     print(continue_logits[~mask_cp])
        #     input()
        # if we are not training, calculate mean step accuracy
        # for outputs/logits/predictions
        mean_accs = type(self).calculate_step_acc(type(self).get_outputs(output_logits_masked), output_real_masked, masked_batch)
        self.mean_step.extend(mean_accs)
        term_correct_accs = type(self).calculate_step_acc((continue_logits > 0).float(), true_termination, torch.unique(batch.batch))
        self.term_preds_mean_step.extend(term_correct_accs)
        if (mean_accs != 1).any() or (term_correct_accs != 1).any():
            self.wrong_flag = True


    def get_step_loss(self,
                      mask,
                      mask_cp,
                      y_curr,
                      output_logits,
                      true_termination,
                      continue_logits,
                      compute_losses=True):

        # Take prediction (logits for outputs) and
        # target values (real for outputs) for
        # the graphs that still proceed with execution
        output_logits_masked = output_logits[mask]
        output_real_masked = y_curr[mask].float()

        # Accumulate number of steps done. Each unfrozen graph contributes with 1 step.
        steps = sum(mask_cp.float())

        loss_output, loss_term, processed_nodes = 0, 0, 0

        # If we simply want to execute (e.g. when testing displaying), we drop
        # losses calculation
        if compute_losses:
            processed_nodes = len(output_real_masked)
            loss_output = self.get_output_loss(output_logits_masked, output_real_masked)

            # calculate losses for termination from masked out graphs.
            # NOTE we use sum reduction as we will do the averaging later
            # (That's why we have sum of steps and sum of nodes)
            loss_term = F.binary_cross_entropy_with_logits(
                continue_logits[mask_cp],
                true_termination[mask_cp].float(),
                reduction='sum')
            if get_hyperparameters()['calculate_termination_statistics']:
                self.update_termination_statistics(continue_logits[mask_cp], true_termination[mask_cp].float())


        return loss_output, loss_term, processed_nodes

    def aggregate_steps(self, steps, processed_nodes):
        self.sum_of_steps += steps
        self.sum_of_processed_nodes += processed_nodes
        if not self.training:
            self.validation_sum_of_processed_nodes += processed_nodes
            self.validation_sum_of_steps += steps

    def aggregate_loss_steps_and_acc(self,
                                     batch,
                                     mask,
                                     mask_cp,
                                     y_curr,
                                     output_logits,
                                     true_termination,
                                     continue_logits,
                                     compute_losses=True):

        loss_output, loss_term, processed_nodes =\
                self.get_step_loss(
                    mask, mask_cp,
                    y_curr, output_logits,
                    true_termination, continue_logits,
                    compute_losses=compute_losses)

        self.losses['total_loss_output'] += loss_output
        self.losses['total_loss_term'] += loss_term

        if not self.training:
            self.aggregate_step_acc(batch, mask, mask_cp, y_curr, output_logits,
                                    true_termination, continue_logits)

        steps = sum(mask_cp.float())
        self.aggregate_steps(steps, processed_nodes)

    def aggregate_last_step(self, batch_ids, output, real):
        last_step_accs = type(self).calculate_step_acc(type(self).get_outputs(output), real, batch_ids)
        if (last_step_accs != 1).any():
            self.wrong_flag = True
        self.last_step.extend(last_step_accs)
        self.last_step_total += len(last_step_accs)


    def prepare_constants(self, batch):
        SIZE = batch.num_nodes
        # we make at most |V|-1 steps
        GRAPH_SIZES = torch_scatter.scatter(torch.ones_like(batch.batch), batch.batch, reduce='sum')
        STEPS_SIZE = GRAPH_SIZES.max()
        return SIZE, STEPS_SIZE

    def set_initial_last_states(self, batch):
        self.last_latent = torch.zeros(batch.num_nodes, _DIM_LATENT, device=_DEVICE)
        self.last_continue_logits = torch.ones(batch.num_graphs, device=_DEVICE)

        self.last_output_logits = torch.where(batch.x[0, :, 1].bool().unsqueeze(-1), 1e3, -1e3)
        self.last_output = (self.last_output_logits > 0).float()

    def update_states(self, current_latent,
                      output_logits, continue_logits):
        def update_per_mask(before, after, mask=None):
            # NOTE: this does expansion of the mask, if you do
            # NOT use expansion, use torch.where
            if mask is None:
                mask = self.mask
            mask = mask.unsqueeze(-1).expand_as(before)
            return torch.where(mask, after, before)
        self.last_continue_logits = torch.where(self.mask_cp, continue_logits,
                                                self.last_continue_logits)
        self.last_latent = update_per_mask(self.last_latent, current_latent)
        self.last_output_logits = update_per_mask(self.last_output_logits, output_logits)
        self.last_output = type(self).get_outputs(self.last_output_logits).float()

    def prepare_initial_masks(self, batch):
        self.mask = torch.ones_like(batch.batch, dtype=torch.bool, device=_DEVICE)
        self.mask_cp = torch.ones(batch.num_graphs, dtype=torch.bool, device=_DEVICE)
        self.edge_mask = torch.ones_like(batch.edge_index[0], dtype=torch.bool, device=_DEVICE)

    def get_losses_dict(self, validation=False):
        # NOTE Here we do the averaging. The sum (not sum of mean-reduced losses!!!)
        # is averaged over the sum of steps (for termination outputs/logits) or the sum of
        # all nodes ever processed (for termination outputs/logits)

        # NOTE 2, note that for training, these losses are average per-batch, whereas
        # for validation, these losses are averaged over the whole val/testing set.

        if self.hardcode_outputs:
            outmul = 0
        else:
            outmul = 1

        if validation:
            losses_dict = {
                'total_loss_output': self.lambda_mul*outmul*self.losses['total_loss_output'] / (self.validation_sum_of_processed_nodes * self.output_features),
                'total_loss_term': 1*self.lambda_mul*1*self.losses['total_loss_term'] / self.validation_sum_of_steps,
            }  if self.validation_sum_of_processed_nodes else 0
        else:
            losses_dict = {
                'total_loss_output': self.lambda_mul*outmul* self.losses['total_loss_output'] / (self.sum_of_processed_nodes * self.output_features),
                'total_loss_term': self.lambda_mul*1*self.losses['total_loss_term'] / self.sum_of_steps,
            } if self.sum_of_processed_nodes else 0

        return losses_dict


    def get_training_loss(self):
        return sum(self.get_losses_dict().values()) if self.get_losses_dict() != 0 else 0

    def get_validation_losses(self):
        return sum(self.get_losses_dict(validation=True).values()) if self.get_losses_dict(validation=True) != 0 else 0

    def get_validation_accuracies(self):
        assert not torch.isnan(torch.tensor(self.mean_step)).any(), torch.tensor(self.mean_step)
        assert self.last_step_total == len(self.last_step)
        return {
            'mean_step_acc': torch.tensor(self.mean_step).sum()/len(self.mean_step),
            'term_mean_step_acc': torch.tensor(self.term_preds_mean_step).sum()/(len(self.term_preds_mean_step) if self.term_preds_mean_step else 1), # to avoid div by 0
            'last_step_acc': torch.tensor(self.last_step).mean(),
        }

    def zero_hidden(self, num_nodes):
        self.hidden = torch.zeros(num_nodes, self.latent_features).to(get_hyperparameters()['device'])

    def loop_condition(self, termination, STEPS_SIZE):
        return (((not self.training and termination.any()) or
                 (self.training and termination.any())) and
                 self.step_idx < STEPS_SIZE-1)

    def loop_body(self,
                  batch,
                  inp,
                  y_curr,
                  true_termination,
                  compute_losses):

        current_latent, output_logits, continue_logits =\
            self(
                batch,
                inp,
                batch.edge_index
            )

        termination = continue_logits

        self.debug_batch = batch
        self.debug_y_curr = y_curr
        self.update_states(current_latent,
                           output_logits, termination)

        self.aggregate_loss_steps_and_acc(
            batch, self.mask, self.mask_cp,
            y_curr, output_logits,
            true_termination, continue_logits,
            compute_losses=compute_losses)

    def get_input_from_output(self, output, batch=None):
        output = (output.long() > 0)
        return F.one_hot(output.long().squeeze(-1), num_classes=self.node_features).float()

    def get_step_output(self, batch, step):
        output_logits = torch.where(batch.y[step, :, 1].bool().unsqueeze(-1), 1e3, -1e3)
        output = (output_logits > 0).float()
        return output_logits, output

    def get_step_input(self, x_curr, batch):
        return x_curr if (self.training and self.use_TF) else self.get_input_from_output(self.last_output_logits, batch)

    def process(
            self,
            batch,
            EPSILON=0,
            enforced_mask=None,
            compute_losses=True,
            hardcode_outputs=False,
            debug=False,
            **kwargs):
        '''
        Method that takes a batch, does all iterations of every graph inside it
        and accumulates all metrics/losses.
        '''

        SIZE, STEPS_SIZE = self.prepare_constants(batch)
        self.hardcode_outputs = hardcode_outputs

        # Pytorch Geometric batches along the node dimension, but we execute
        # along the temporal (step) dimension, hence we need to transpose
        # a few tensors. Done by `prepare_batch`.
        batch = utils.prepare_batch(batch)
        # When we want to calculate last step metrics/accuracies
        # we need to take into account again different termination per graph
        # hence we save last step tensors (e.g. outputs) into their
        # corresponding tensor. The function below prepares these tensors
        # (all set to zeros, except masking for computation, which are ones)
        self.set_initial_last_states(batch)
        # Prepare masking tensors (each graph does at least 1 iteration of the algo)
        self.prepare_initial_masks(batch)
        # A flag if we had a wrong graph in the batch. Used for visualisation
        # of what went wrong
        self.wrong_flag = False
        assert self.mask_cp.all(), self.mask_cp
        to_process = torch.ones([batch.num_graphs], device=_DEVICE)

        while True:
            # take inputs/target values
            x_curr, y_curr = batch.x[self.step_idx], batch.y[self.step_idx]
            if not self.training:
                assert (self.last_continue_logits > 0).any() or True

            # Some algorithms, e.g. parallel colouring outputs fewer values than it takes
            # (e.g. priorities for colouring are unchanged one every step)
            # so if we reuse our last step outputs, they need to be fed back in.
            # NOTE self.get_step_input always takes x_curr, if we train
            inp = self.get_step_input(x_curr, batch)

            true_termination = batch.termination[self.step_idx] if self.step_idx < STEPS_SIZE else torch.zeros_like(batch.termination[-1])

            # Does one iteration of the algo and accumulates statistics
            self.loop_body(batch,
                           inp,
                           y_curr,
                           true_termination,
                           compute_losses)
            # And calculate what graphs would execute on the next step.
            self.mask, self.mask_cp = type(self).get_masks(self.training, batch, true_termination if self.training else self.last_continue_logits, enforced_mask)
            if not self.loop_condition(
                    batch.termination[self.step_idx] if self.training else self.mask_cp,
                    STEPS_SIZE):
                break
            self.step_idx += 1

        if not self.training:
            # After we are done with the execution, accumulate statistics related
            # to last step accuracies.

            self.aggregate_last_step(
                batch.batch,
                self.last_output_logits,
                batch.y[-1])

    @staticmethod
    def get_outputs(outputs):
        return outputs > 0

    def compute_outputs(self, batch, encoded_nodes, hidden):

        output_logits = self.decoder_network(torch.cat((encoded_nodes, hidden), dim=1))

        if self.hardcode_outputs:
            if type(self) == models.AlgorithmColoring:
                output_logits = torch.where(F.one_hot(batch.y[self.step_idx], num_classes=self.output_features).bool(), 1e3, -1e3).float()
            else:
                output_logits = torch.where(batch.y[self.step_idx].bool(), 1e3, -1e3).unsqueeze(-1)
        return output_logits

    def encode_nodes(self, current_input, last_latent):
        return self.node_encoder(torch.cat((current_input, last_latent), dim=1))

    def forward(self, batch, current_input, edge_index):
        batch_ids = batch.batch

        assert not torch.isnan(self.last_latent).any()
        assert not torch.isnan(current_input).any()
        encoded_nodes = self.encode_nodes(current_input, self.last_latent)
        hidden = self.processor(encoded_nodes, edge_index, self.last_latent)
        assert not torch.isnan(hidden).any()
        assert not torch.isnan(encoded_nodes).any()
        output_logits = self.compute_outputs(batch, encoded_nodes, hidden)
        assert not torch.isnan(output_logits).any(), hidden[torch.isnan(output_logits).squeeze()]
        if self.next_step_pool:
            inp2 = self.get_input_from_output(output_logits, batch=batch)

            encoded_nodes2 = self.encode_nodes(inp2, hidden)
            hidden2 = self.processor(encoded_nodes2, edge_index, hidden)
            catted = hidden2

        if not self.next_step_pool:
            catted = hidden

        continue_logits = self.get_continue_logits(
            batch_ids,
            catted,
            sth_else=None)
        return hidden, output_logits, continue_logits
