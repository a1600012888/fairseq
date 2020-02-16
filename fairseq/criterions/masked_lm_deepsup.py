# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('masked_lm_deepsup')
class MaskedLmDeepsupLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super(MaskedLmDeepsupLoss, self).__init__(args, task)
        self.deepsup_lambda = args.deepsup_lambda

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        super(MaskedLmDeepsupLoss,
              MaskedLmDeepsupLoss).add_args(parser)

        parser.add_argument('--deepsup_lambda', default=0.5, type=float, metavar='D',
                            help='weight for the deeply supervised loss')


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        masked_tokens = sample['target'].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if sample_size == 0:
            masked_tokens = None

        logits, extra = model(**sample['net_input'], masked_tokens=masked_tokens, deepsup=True)
        extra_logits = extra['mid_pred']
        targets = model.get_targets(sample, [logits])

        #import IPython
        #IPython.embed()
        if sample_size != 0:
            targets = targets[masked_tokens]

        loss = F.nll_loss(
            F.log_softmax(
                logits.view(-1, logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )

        mid_loss = F.nll_loss(
            F.log_softmax(
                extra_logits.view(-1, logits.size(-1)),
                dim=-1,
                dtype=torch.float32,
            ),
            targets.view(-1),
            reduction='sum',
            ignore_index=self.padding_idx,
        )
        total_loss = loss + mid_loss * self.deepsup_lambda

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'mid_loss': utils.item(mid_loss.data) if reduce else mid_loss.data,
        }
        return total_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        mid_loss_sum = sum(log.get('mid_loss', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('mid_loss', mid_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: round(2**meters['loss'].avg, 3))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
