# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


@register_criterion('mask_leaner')
class MaskLeanerLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, args, task):
        super(MaskLeanerLoss, self).__init__(args, task)

        self.vocab = self.task.source_dictionary
        self.mask_idx = self.task.mask_idx
        self.mask_prob = self.task.args.mask_prob
        self.leave_unmasked_prob = self.task.args.leave_unmasked_prob
        self.random_token_prob = self.task.args.random_token_prob
        self.rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob

        self.mask_whole_words = self.task.args.mask_whole_words
        self.freq_weighted_replacement = self.task.args.freq_weighted_replacement

        if self.random_token_prob > 0.0:
            if self.freq_weighted_replacement:
                weights = np.array(self.vocab.count)
            else:
                weights = np.ones(len(self.vocab))
            weights[:self.vocab.nspecial] = 0
            self.weights = weights / weights.sum()

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        # compute MLM loss
        # model.learner model.lm
        raw_inps = sample["net_input"]["src_tokens"]
        raw_targets = sample['target']
        raw_masked_tokens = raw_targets.ne(self.padding_idx)
        inps = raw_targets * raw_masked_tokens + \
               raw_inps * (raw_masked_tokens ^ True)
        sz = inps.size(-1) # all batches should be the same length
        num_mask = int(sz * 0.15)

        masker_out = model.masker(inps)[0]#.view(inps.size(0), -1)

        #print('masker 1 shape', masker_out.shape)

        masked_tokens, masked_idxes = torch.topk(masker_out,
                                                 num_mask, dim=-1)

        labels_list = []
        with torch.no_grad():

            #labels = torch.full_like(inps, self.padding_idx)
            #labels[masked_idxes] = inps[masked_idxes]


            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob

            new_inps = []

            #import IPython
            #IPython.embed()

            for i in range(inps.size(0)):
                inp = inps[i]
                mask = torch.full_like(inp, False).type(torch.bool)
                mask[masked_idxes[i]] = True

                label = torch.full_like(inp, self.padding_idx)
                label[masked_idxes[i]] = inp[masked_idxes[i]]
                labels_list.append(label)

                #import IPython
                #IPython.embed()

                if rand_or_unmask_prob > 0.0:
                    tmp_rand = torch.rand_like(inp.type(torch.float))
                    tmp_rand = (tmp_rand < rand_or_unmask_prob)
                    #tmp_rand = tmp_rand.to(inp.device)
                    #tmp_rand = (torch.rand(sz) < rand_or_unmask_prob).to(mask.device)
                    tmp_rand = tmp_rand.type(mask.type())
                    rand_or_unmask = mask & tmp_rand
                    if self.random_token_prob == 0.0:
                        unmask = rand_or_unmask
                        rand_mask = None
                    elif self.leave_unmasked_prob == 0.0:
                        unmask = None
                        rand_mask = rand_or_unmask
                    else:
                        unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                        decision = torch.rand_like(inp.type(torch.float))  < unmask_prob
                        decision = decision.type(mask.type())
                        unmask = rand_or_unmask & decision
                        rand_mask = rand_or_unmask & (~decision)
                else:
                    unmask = rand_mask = None

                if unmask is not None:
                    mask = mask ^ unmask

                #if self.mask_whole_words is not None:
                #    #mask = torch.repeat(mask, word_lens)
                #    mask = mask.repeat(word_lens)



                new_item = inp.clone()
                new_item[mask] = self.mask_idx
                if rand_mask is not None:
                    num_rand = rand_mask.sum()
                    if num_rand > 0:
                        #if self.mask_whole_words is not None:
                        #    #rand_mask = torch.repeat(rand_mask, word_lens)
                        #    rand_mask = rand_mask.repeat(word_lens)
                        #    num_rand = rand_mask.sum()
                        #import IPython
                        #IPython.embed()
                        rand_tensor = torch.tensor(
                            np.random.choice(len(self.vocab),
                                             num_rand.cpu().numpy(),
                                             p=self.weights)).to(mask.device)
                        rand_tensor.type(inps.type())
                        new_item[rand_mask] = rand_tensor
                new_inps.append(new_item)

            new_inp = torch.stack(new_inps, dim=0)
            labels = torch.stack(labels_list, dim=0)

            sample['target'] = labels
            sample['net_input']["src_tokens"] = new_inp
        masked_tokens = sample['target'].ne(self.padding_idx)
        sample_size = masked_tokens.int().sum().item()

        # (Rare case) When all tokens are masked, the model results in empty
        # tensor and gives CUDA error.
        if sample_size == 0:
            masked_tokens = None

        logits = model(**sample['net_input'], masked_tokens=masked_tokens)[0]
        targets = model.get_targets(sample, [logits])

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

        pred = torch.argmax(logits, dim=-1)
        is_right = (pred == targets).type(loss.type())

        masker_fail_table = torch.full_like(inps, 0).type(loss.type())
        masker_fail_table[masked_tokens] = is_right
        masker_loss = masker_fail_table * masker_out
        masker_loss = masker_loss.sum()

        total_loss = masker_loss * 0.1 + loss

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['nsentences'],
            'sample_size': sample_size,
            'masker_loss':utils.item(masker_loss.data) if reduce else masker_loss.data,
            'total_loss': utils.item(total_loss.data) if reduce else total_loss.data,
        }
        return total_loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        masker_loss_sum = sum(log.get('masker_loss', 0) for log in logging_outputs)
        total_loss_sum = sum(log.get('total_loss', 0) for log in logging_outputs)

        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('masker_loss', masker_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('total_loss', total_loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived('ppl', lambda meters: round(2**meters['loss'].avg, 3))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
