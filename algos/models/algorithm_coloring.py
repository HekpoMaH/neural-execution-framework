import torch
from overrides import overrides
import torch.nn as nn
import torch.nn.functional as F
from algos.models import AlgorithmBase

class AlgorithmColoring(AlgorithmBase):
    '''
    The overriding in this class comes from the fact that (only) the parallel
    coloring ouptuts are not the same dimension as its inputs: The algorithm
    takes as input priorities, in addition to current coloring on the last step
    but is not required to output these priorities.
    '''

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
                 **kwargs):

        super(AlgorithmColoring, self).__init__(
            latent_features,
            node_features,
            output_features,
            algo_processor,
            dataset_class,
            inside_class,
            dataset_root,
            dataset_kwargs,
            bias=bias,
            **kwargs)
        self.bit_encoder = nn.Sequential(
                nn.Linear(node_features - output_features, latent_features), # NOTE I'm using the fact that bits are not part of the output features
                nn.LeakyReLU(),
                nn.Linear(latent_features, latent_features), # NOTE I'm using the fact that bits are not part of the output features
                nn.LeakyReLU()
        )
        self.color_encoder = nn.Sequential(
                nn.Linear(output_features, latent_features), # NOTE Output features = colors we have+1
                nn.LeakyReLU()
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(3*latent_features, latent_features, bias=bias),
            nn.LeakyReLU(),
            nn.Linear(latent_features, latent_features, bias=bias),
            nn.LeakyReLU()
        )

    @overrides
    def encode_nodes(self, inp, last_latent):
        def bin2dec(b, bits):
            mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
            return torch.sum(mask * b, -1)
        colors = inp[:, :self.output_features]
        bits = inp[:, self.output_features:] # NOTE Again, as above, bits are not parts of output
        encoded_colors = self.color_encoder(colors)
        encoded_bits = self.bit_encoder(bits)
        inp = torch.cat((encoded_colors, encoded_bits), dim=-1)
        return self.node_encoder(torch.cat((inp, last_latent), dim=-1))

    @overrides
    def get_input_from_output(self, output, batch=None):
        output = type(self).get_outputs(output)
        return torch.cat((F.one_hot(output.long().squeeze(-1), num_classes=self.output_features).float(), batch.priorities), dim=-1)

    @overrides
    def get_output_loss(self, output_logits, target):
        assert F.cross_entropy(output_logits, target.long(), reduction='sum') >= 0
        return F.cross_entropy(output_logits, target.long(), reduction='sum', weight=getattr(self.dataset, 'class_weights', None)) #FIXME remove the s-es if you want them back

    @overrides
    def get_step_output(self, batch, step):
        output_logits = torch.where(F.one_hot(batch.y[step, :], self.output_features).bool(), 1e3, -1e3)
        output = (output_logits > 0).float()
        return output_logits, output

    @overrides
    def set_initial_last_states(self, batch):
        super().set_initial_last_states(batch)
        # the parent method initialises the initial last output logits
        # from the input logits. The output logits should instead
        # be 1+|number of colours|
        self.last_output_logits = torch.where(batch.x[0, :, :self.output_features].bool(), 1e3, -1e3)
        self.last_output = (self.last_output_logits > 0).float()

    @staticmethod
    @overrides
    def get_outputs(outputs):
        return outputs.argmax(dim=-1)

    @overrides
    def get_losses_dict(self, validation=False):
        # NOTE Here we do the averaging. The sum (not sum of mean-reduced losses!!!)
        # is averaged over the sum of steps (for termination outputs/logits) or the sum of
        # all nodes ever processed (for termination outputs/logits)

        # NOTE 2, note that for training, these losses are average per-batch, whereas
        # for validation, these losses are averaged over the whole val/testing set.

        term_mul = 0.05
        if validation:
            losses_dict = {
                'total_loss_output': self.lambda_mul*self.losses['total_loss_output'] / (self.validation_sum_of_processed_nodes * self.output_features),
                'total_loss_term': term_mul*self.lambda_mul*1*self.losses['total_loss_term'] / self.validation_sum_of_steps,
            }  if self.validation_sum_of_processed_nodes else 0
        else:
            losses_dict = {
                'total_loss_output': self.lambda_mul*self.losses['total_loss_output'] / (self.sum_of_processed_nodes * self.output_features),
                'total_loss_term': term_mul*self.lambda_mul*1*self.losses['total_loss_term'] / self.sum_of_steps,
            } if self.sum_of_processed_nodes else 0

        return losses_dict

if __name__ == '__main__':
    print(list(dir(AlgorithmColoring)))
