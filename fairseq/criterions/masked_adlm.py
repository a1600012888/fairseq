# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('masked_adlm')
class MaskedAdLmLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """


    def __init__(self, args, task):
        super(MaskedAdLmLoss, self).__init__(args, task)

        self.vocab = self.task.source_dictionary
        print(len(self.vocab.count))
        self.register_buffer('margins', torch.zeros((len(self.vocab.count), 1)))
        self.margins.requires_grad = False

        self.margin_lambda = args.margin_lambda
        self.margin_lr = args.margin_lr
        self.margin_norm = args.margin_norm

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        super(MaskedAdLmLoss,
              MaskedAdLmLoss).add_args(parser)
        parser.add_argument('--margin_lambda', default=0.5, type=float, metavar='D',
                            help='weight for the adaptive margin loss')
        parser.add_argument('--margin_lr', default=0.0001, type=float, metavar='D',
                            help='weight for the adaptive margin loss')
        parser.add_argument('--margin-norm', default='l1', type=str,
                            help='Type of margin norm in the loss')

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        #self.margins.requires_grad = model.training

        masked_tokens = sample['target'].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if sample_size == 0:
            masked_tokens = None

        logits = model(**sample['net_input'], masked_tokens=masked_tokens)[0]
        targets = model.get_targets(sample, [logits])

        #import IPython
        #IPython.embed()
        if sample_size != 0:
            targets = targets[masked_tokens]


        # targets shape: [x]
        # logits.shape: [x, 32769]
        one_hot = F.one_hot(targets, len(self.vocab.count)) # [x, 32769]

        #import IPython
        #IPython.embed()

        m = F.embedding(targets, self.margins)  # [x, 1]
        #m = self.margins(targets).squeeze(dim=-1)
        margin = m * one_hot # [x, 32769]

        #import IPython
        #IPython.embed()

        logits_minus_margin = logits - margin
        log_softmax = F.log_softmax(
                logits_minus_margin.view(-1, logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ) # [x, 32769]


        adm_loss = F.nll_loss(
            log_softmax, 
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        # cal margin grad
        with torch.no_grad():
            margin_log_grad = torch.gather(log_softmax.detach(), dim=-1,
                                           index=targets.unsqueeze(-1)) # [x, 1]
            margin_grad_cross = torch.exp(margin_log_grad) - \
                                torch.ones_like(margin_log_grad)

            if self.margin_norm == 'l1':
                margin_grad = margin_grad_cross + torch.ones_like(m) * self.margin_lambda
            else:
                # l2 norm
                margin_grad = margin_grad_cross + m * self.margin_lambda * 2.0
            margin_update = -1.0 * margin_grad *  self.margin_lr

            self.margins.scatter_add_(0, targets.unsqueeze(-1), margin_update.half())

            # for logging below!  margin_norm;  normal loss
            margin_norm = torch.mean(self.margins) * sample['nsentences']# used for log!

            normal_loss = F.nll_loss(
                F.log_softmax(
                    logits.view(-1, logits.size(-1)),
                    dim=-1,
                    dtype=torch.float32,
                ),
                targets.view(-1),
                reduction='sum',
                ignore_index=self.padding_idx,
            )

        logging_output = {
            'loss': utils.item(normal_loss.data) if reduce else normal_loss.data,
            'margin_n':utils.item(margin_norm.data) if reduce else margin_norm.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'admloss': utils.item(adm_loss.data) if reduce else adm_loss.data,
        }
        return adm_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        admloss_sum = sum(log.get('admloss', 0) for log in logging_outputs)
        margin_n = sum(log.get('margin_n', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('admloss', admloss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('margin_norm', margin_n / nsentences, 32, round=3)
        metrics.log_derived('ppl', lambda meters: round(2**meters['loss'].avg, 3))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
