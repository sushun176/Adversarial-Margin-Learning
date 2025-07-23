import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):

    def __init__(self, alpha, temp):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.temp = temp

    def nt_xent_loss(self, anchor, target, labels):
        with torch.no_grad():
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1))
            # delete diag elem
            mask = mask ^ torch.diag_embed(torch.diag(mask))
        # compute logits
        anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp
        # delete diag elem
        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()
        # compute log prob
        exp_logits = torch.exp(logits)
        # mask out positives
        logits = logits * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        # in case that mask.sum(1) is zero
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        # compute log-likelihood
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = -1 * pos_logits.mean()
        return loss

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets)
        cl_loss = self.alpha * self.nt_xent_loss(normed_cls_feats, normed_cls_feats, targets)
        return ce_loss + cl_loss

def adversarial_margin_loss(h_adv, label_embed, y_true, margin=0.3):
    # h_adv: [B, H], label_embed: [C, H]
    sim_adv = F.cosine_similarity(h_adv.unsqueeze(1), label_embed.unsqueeze(0), dim=-1)  # [B, C]

    pos_sim = sim_adv.gather(1, y_true.unsqueeze(1)).squeeze(1)  # [B]

    mask = torch.ones_like(sim_adv).bool()
    mask.scatter_(1, y_true.unsqueeze(1), False)
    sim_neg = sim_adv.masked_fill(~mask, -1e9)
    neg_sim, _ = sim_neg.max(dim=1)  # [B]

    loss = F.relu(neg_sim - pos_sim + margin).mean()
    return loss

# def adversarial_pairwise_margin_loss(h_adv, label_embed, y_true, margin=0.3):
#
#     sim_adv = F.cosine_similarity(h_adv.unsqueeze(1), label_embed.unsqueeze(0), dim=-1)  # [B, C]
#     pos_sim = sim_adv.gather(1, y_true.unsqueeze(1))  # [B, 1]
#     pairwise_diff = sim_adv - pos_sim
#     loss_terms = F.relu(pairwise_diff + margin) # [B, C]
#     mask = torch.ones_like(sim_adv).bool() # [B, C]
#     mask.scatter_(1, y_true.unsqueeze(1), False) # 在真实标签的索引位置设置为 False
#     loss = loss_terms.masked_select(mask).mean()
#     return loss
class AMLLoss():

    def __init__(self):
        super().__init__()

    def  __call__(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)

        normed_label_feats = F.normalize(outputs['label_feats'], dim=-1)
        normed_pos_label_feats = torch.gather(normed_label_feats, dim=1, index=targets.reshape(-1, 1, 1).expand(-1, 1, normed_label_feats.size(-1))).squeeze(1)
        aml_loss = adversarial_margin_loss(normed_cls_feats,normed_pos_label_feats,targets)
  #      ce_loss = F.cross_entropy(outputs['predicts'], targets)
        return aml_loss