import torch
from torch import nn
from transformers.modeling_electra import ElectraPreTrainedModel, ElectraModel
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss, L1Loss
import  torch.nn.functional as F


def masked_softmax(vector, mask, dim=-1, memory_efficient=False, mask_fill_value=-1e32):
    if mask is None:
        result = torch.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            result = torch.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).to(dtype=torch.bool), mask_fill_value)
            result = torch.nn.functional.softmax(masked_vector, dim=dim)
    return result


def weighted_sum(matrix, attention):
    if attention.dim() == 2 and matrix.dim() == 3:
        return attention.unsqueeze(1).bmm(matrix).squeeze(1)
    if attention.dim() == 3 and matrix.dim() == 3:
        return attention.bmm(matrix)
    if matrix.dim() - 1 < attention.dim():
        expanded_size = list(matrix.size())
        for i in range(attention.dim() - matrix.dim() + 1):
            matrix = matrix.unsqueeze(1)
            expanded_size.insert(i + 1, attention.size(i + 1))
        matrix = matrix.expand(*expanded_size)
    intermediate = attention.unsqueeze(-1).expand_as(matrix) * matrix
    return intermediate.sum(dim=-2)


class ElectraDependencyModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dep_label_classifier = nn.Linear(2 * config.hidden_size, 2)

        self.init_weights()

    def forward(self, input_ids, attention, child, head, mask_entail, mask_cont,
                sent_label, num_dependency, device='cuda', weights=None, **kwargs):
        batch_size = input_ids.size(0)

        transformer_outputs = self.electra(input_ids, attention_mask=attention)

        outputs = transformer_outputs[0]
        outputs = self.dropout(outputs)
        padding_value = torch.zeros(batch_size, 1, outputs.size(-1)).to(device)
        outputs = torch.cat((outputs, padding_value), 1)

        child = child.squeeze(2)
        head = head.squeeze(2)

        add = torch.arange(batch_size) *  outputs.size(1)
        add = add.unsqueeze(1).to(device)
        child_temp = child + add
        head_temp = head + add
        outputs = outputs.view((-1, outputs.size(-1)))
        child_embeddings = outputs[child_temp]
        head_embeddings = outputs[head_temp]

        child_embeddings = child_embeddings.view(batch_size, -1, child_embeddings.size(-1))
        head_embeddings = head_embeddings.view(batch_size, -1, head_embeddings.size(-1))

        final_embeddings = torch.cat([child_embeddings, head_embeddings], dim=2)
        logits_all = self.dep_label_classifier(final_embeddings)
        
        mask = torch.arange(mask_entail.size(1)).to(device)[None, :] >= num_dependency[:, None]
        mask = mask.type(torch.long) * -1
        mask_entail = mask_entail + mask
        mask_entail = mask_entail.detach()

        loss_fct_dep = CrossEntropyLoss(ignore_index=-1, weight=weights)
        loss = loss_fct_dep(logits_all.view(-1, 2), mask_entail.view(-1).type(torch.long))

        outputs_return = (logits_all,)
        outputs_return = (loss,) + outputs_return

        return outputs_return


class SumPhraseModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.span_attention_logits_layer = nn.Linear(config.hidden_size, 1)

        self.phrase_rel_linear = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.phrase_linear = nn.Linear(config.hidden_size, config.hidden_size)
        
        self.gelu = nn.GELU()

        self.phrase_label_classifier = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(self, input_ids, attention, child, head, mask_entail, mask_cont,
                sent_label, num_dependency, device='cuda', weights=None, **kwargs):

        batch_size = input_ids.size(0)
        transformer_outputs = self.electra(input_ids, attention_mask=attention)

        outputs = transformer_outputs[0]
        outputs = self.dropout(outputs)

        padding_value = torch.zeros(batch_size, 1, outputs.size(-1)).to(device)
        outputs = torch.cat((outputs, padding_value), 1)

        # span
        padding_id = outputs.size(1) - 1
        head_mask = head != padding_id
        child_mask = child != padding_id

        add = torch.arange(batch_size) * outputs.size(1)
        add = add.unsqueeze(1).to(device)

        outputs = outputs.view((-1, outputs.size(-1)))

        #span
        token_attention_logits = self.span_attention_logits_layer(outputs)

        max_phrase = child.size(1)
        child_temp = child.view(batch_size, -1) + add
        head_temp = head.view(batch_size, -1) + add

        child_embeddings = outputs[child_temp]
        head_embeddings = outputs[head_temp]

        child_embeddings = child_embeddings.view(batch_size, max_phrase, -1, child_embeddings.size(-1))
        head_embeddings = head_embeddings.view(batch_size, max_phrase, -1, head_embeddings.size(-1))
        
        
        # span
        head_logits = token_attention_logits[head_temp]
        child_logits = token_attention_logits[child_temp]

        head_logits = head_logits.view(batch_size, max_phrase, -1)
        child_logits = child_logits.view(batch_size, max_phrase, -1)

        head_weight = masked_softmax(head_logits, head_mask)
        head_embeddings = weighted_sum(head_embeddings, head_weight)

        child_weight = masked_softmax(child_logits, child_mask)
        child_embeddings = weighted_sum(child_embeddings, child_weight)

        same_head_child = torch.all(child == head, dim=2)
        same_head_child_embeddings = child_embeddings[same_head_child]
        same_head_child_embeddings = self.gelu(self.phrase_linear(same_head_child_embeddings))

        not_same_head_child_embeddings = torch.cat([child_embeddings, head_embeddings], dim=2)
        final_embeddings = self.gelu(self.phrase_rel_linear(not_same_head_child_embeddings))
        final_embeddings[same_head_child] = same_head_child_embeddings

        logits_all = self.phrase_label_classifier(final_embeddings)

        mask = torch.arange(mask_entail.size(1)).to(device)[None, :] >= num_dependency[:, None]
        mask = mask.type(torch.long) * -1
        mask_entail = mask_entail + mask
        mask_entail = mask_entail.detach()

        loss_fct_dep = CrossEntropyLoss(ignore_index=-1, weight=weights)
        loss = loss_fct_dep(logits_all.view(-1, 2), mask_entail.view(-1).type(torch.long))

        outputs_return = (logits_all,)
        outputs_return = (loss,) + outputs_return

        return outputs_return
    
    
class SumPhraseMultiTaskModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        self.span_attention_logits_layer = nn.Linear(config.hidden_size, 1)

        self.phrase_rel_linear = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.phrase_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        
        self.phrase_label_classifier = nn.Linear(config.hidden_size, 2)
        self.sent_basis_linear = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.sent_basis_label_classifier = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(self, input_ids, attention, child, head, mask_entail, mask_cont,
                sent_label, num_dependency,  hypo_cls_idx, sent_indices, sent_basis_label, 
                device='cuda', weights=None, multitask_loss_weight=[1,1], **kwargs):
        batch_size = input_ids.size(0)
        
        transformer_outputs = self.electra(input_ids, attention_mask=attention)

        outputs = transformer_outputs[0]
        outputs = self.dropout(outputs)

        padding_value = torch.zeros(batch_size, 1, outputs.size(-1)).to(device)
        outputs = torch.cat((outputs, padding_value), 1)

        # span
        padding_id = outputs.size(1) - 1
        head_mask = head != padding_id
        child_mask = child != padding_id
        sent_mask = sent_indices != padding_id
        
        add = torch.arange(batch_size) * outputs.size(1)
        add = add.unsqueeze(1).to(device)

        outputs = outputs.view((-1, outputs.size(-1)))

        #span
        token_attention_logits = self.span_attention_logits_layer(outputs)
        
        max_phrase = child.size(1)
        max_num_sent = sent_indices.size(1)
        
        child_temp = child.view(batch_size, -1) + add
        head_temp = head.view(batch_size, -1) + add
        hypo_cls_idx += add.squeeze(1)
        sent_indices_tmp = sent_indices.view(batch_size, -1) + add
        
        child_embeddings = outputs[child_temp]
        head_embeddings = outputs[head_temp]
        hypo_cls_embeddings = outputs[hypo_cls_idx]
        sent_embeddings = outputs[sent_indices_tmp]
        
        child_embeddings = child_embeddings.view(batch_size, max_phrase, -1, child_embeddings.size(-1))
        head_embeddings = head_embeddings.view(batch_size, max_phrase, -1, head_embeddings.size(-1))
        sent_embeddings = sent_embeddings.view(batch_size, max_num_sent, -1, sent_embeddings.size(-1))
        
        # span
        head_logits = token_attention_logits[head_temp]
        child_logits = token_attention_logits[child_temp]
        sent_logits = token_attention_logits[sent_indices_tmp]

        head_logits = head_logits.view(batch_size, max_phrase, -1)
        child_logits = child_logits.view(batch_size, max_phrase, -1)
        sent_logits = sent_logits.view(batch_size, max_num_sent, -1)

        head_weight = masked_softmax(head_logits, head_mask)
        head_embeddings = weighted_sum(head_embeddings, head_weight)

        child_weight = masked_softmax(child_logits, child_mask)
        child_embeddings = weighted_sum(child_embeddings, child_weight)
        
        sent_weight = masked_softmax(sent_logits, sent_mask)
        sent_embeddings = weighted_sum(sent_embeddings, sent_weight)
        
        same_head_child = torch.all(child == head, dim=2)
        same_head_child_embeddings = child_embeddings[same_head_child]
        same_head_child_embeddings = self.gelu(self.phrase_linear(same_head_child_embeddings))

        not_same_head_child_embeddings = torch.cat([child_embeddings, head_embeddings], dim=2)
        final_embeddings = self.gelu(self.phrase_rel_linear(not_same_head_child_embeddings))
        final_embeddings[same_head_child] = same_head_child_embeddings
        
        logits_all = self.phrase_label_classifier(final_embeddings)
        
        # hypo
        hypo_cls_embeddings = hypo_cls_embeddings.unsqueeze(1).repeat(1, max_num_sent, 1)
        sent_embeddings = torch.cat([sent_embeddings, hypo_cls_embeddings], dim=2)
        sent_embeddings = self.gelu(self.sent_basis_linear(sent_embeddings))
        sent_basis_logits = self.sent_basis_label_classifier(sent_embeddings)
        
        mask = torch.arange(mask_entail.size(1)).to(device)[None, :] >= num_dependency[:, None]
        mask = mask.type(torch.long) * -1
        mask_entail = mask_entail + mask
        mask_entail = mask_entail.detach()

        loss_fct_dep = CrossEntropyLoss(ignore_index=-1, weight=weights)
        loss_fct_sent = CrossEntropyLoss(ignore_index=-1)
        loss_dep = loss_fct_dep(logits_all.view(-1, 2), mask_entail.view(-1).type(torch.long))
        
        sent_basis_label = sent_basis_label[:, :max_num_sent].contiguous()
        loss_sent = loss_fct_sent(sent_basis_logits.view(-1, 2), sent_basis_label.view(-1))
        
        loss = multitask_loss_weight[0]*loss_dep + multitask_loss_weight[1]*loss_sent
        
        outputs_return = (loss, logits_all, sent_basis_logits, loss_dep, loss_sent)

        return outputs_return


class ElectraDependencyMultiTaskModel(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.electra = ElectraModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dep_label_classifier = nn.Linear(2 * config.hidden_size, 2)

        self.span_attention_logits_layer = nn.Linear(config.hidden_size, 1)
        self.sent_basis_linear = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.sent_basis_label_classifier = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(self, input_ids, attention, child, head, mask_entail, mask_cont,
                sent_label, num_dependency,  hypo_cls_idx, sent_indices, sent_basis_label, 
                device='cuda', weights=None, multitask_loss_weight=[1,1], **kwargs):
        
        batch_size = input_ids.size(0)
        transformer_outputs = self.electra(input_ids, attention_mask=attention)

        outputs = transformer_outputs[0]
        outputs = self.dropout(outputs)
        padding_value = torch.zeros(batch_size, 1, outputs.size(-1)).to(device)
        outputs = torch.cat((outputs, padding_value), 1)

        child = child.squeeze(2)
        head = head.squeeze(2)
        max_num_sent = sent_indices.size(1)

        padding_id = outputs.size(1) - 1
        sent_mask = sent_indices != padding_id

        add = torch.arange(batch_size) *  outputs.size(1)
        add = add.unsqueeze(1).to(device)
        child_temp = child + add
        head_temp = head + add

        hypo_cls_idx += add.squeeze(1)
        sent_indices_tmp = sent_indices.view(batch_size, -1) + add

        outputs = outputs.view((-1, outputs.size(-1)))
        #span
        token_attention_logits = self.span_attention_logits_layer(outputs)

        child_embeddings = outputs[child_temp]
        head_embeddings = outputs[head_temp]
        hypo_cls_embeddings = outputs[hypo_cls_idx]
        sent_embeddings = outputs[sent_indices_tmp]

        child_embeddings = child_embeddings.view(batch_size, -1, child_embeddings.size(-1))
        head_embeddings = head_embeddings.view(batch_size, -1, head_embeddings.size(-1))
        sent_embeddings = sent_embeddings.view(batch_size, max_num_sent, -1, sent_embeddings.size(-1))

        final_embeddings = torch.cat([child_embeddings, head_embeddings], dim=2)
        logits_all = self.dep_label_classifier(final_embeddings)

        sent_logits = token_attention_logits[sent_indices_tmp]
        sent_logits = sent_logits.view(batch_size, max_num_sent, -1)
        sent_weight = masked_softmax(sent_logits, sent_mask)
        sent_embeddings = weighted_sum(sent_embeddings, sent_weight)

        hypo_cls_embeddings = hypo_cls_embeddings.unsqueeze(1).repeat(1, max_num_sent, 1)
        sent_embeddings = torch.cat([sent_embeddings, hypo_cls_embeddings], dim=2)
        sent_embeddings = F.gelu(self.sent_basis_linear(sent_embeddings))
        sent_basis_logits = self.sent_basis_label_classifier(sent_embeddings)
        
        mask = torch.arange(mask_entail.size(1)).to(device)[None, :] >= num_dependency[:, None]
        mask = mask.type(torch.long) * -1
        mask_entail = mask_entail + mask
        mask_entail = mask_entail.detach()

        loss_fct_dep = CrossEntropyLoss(ignore_index=-1, weight=weights)
        loss_fct_sent = CrossEntropyLoss(ignore_index=-1)
        loss_dep = loss_fct_dep(logits_all.view(-1, 2), mask_entail.view(-1).type(torch.long))

        sent_basis_label = sent_basis_label[:, :max_num_sent].contiguous()
        loss_sent = loss_fct_sent(sent_basis_logits.view(-1, 2), sent_basis_label.view(-1))
        
        loss = multitask_loss_weight[0]*loss_dep + multitask_loss_weight[1]*loss_sent
        
        outputs_return = (loss, logits_all, sent_basis_logits, loss_dep, loss_sent)

        return outputs_return
