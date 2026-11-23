import math
from torch.nn import functional as F
from torch import nn
import torch
import numpy as np


class ConvTrans_decoder(torch.nn.Module):
    def __init__(self, num_E, h_dim, device, sequence_len=10, alpha=0.5, d_k=64, d_v=64, n_heads=8,
                 d_ff=2048, n_layers_=1, input_dropout=0, hidden_dropout=0,
                 feature_map_dropout=0, channels=50, kernel_size=3):
        super(ConvTrans_decoder, self).__init__()
        self.num_E = num_E
        self.device = device
        self.alpha = alpha

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.bn0_pre = torch.nn.BatchNorm1d(2)
        self.bn1_pre = torch.nn.BatchNorm1d(channels)
        self.conv1_pre = torch.nn.Conv1d(2, channels, kernel_size, stride=1, padding=int(
            math.floor(kernel_size / 2)))

        self.bn0 = torch.nn.BatchNorm1d(3)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.conv1 = torch.nn.Conv1d(3, channels, kernel_size, stride=1, padding=int(
            math.floor(kernel_size / 2)))

        self.bn2 = torch.nn.BatchNorm1d(h_dim)
        self.fc = torch.nn.Linear(h_dim * channels, h_dim)
        self.layers = nn.ModuleList([DecoderLayer(h_dim, d_k, d_v, n_heads, d_ff, device) for _ in range(n_layers_)])
        self.g_mlp = nn.Linear(h_dim, num_E)
        self.query_context_gate = nn.Linear(h_dim * 2, h_dim)
        self.query_context_norm = nn.LayerNorm(h_dim)
        self.rule_mlp = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(h_dim, num_E),
        )
        self.historical_classifier = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(h_dim, 1),
        )


    def _project_pair(self, subject_state, relation_state, local_state=None):
        inputs = [subject_state, relation_state]
        if local_state is None:
            normalize_inputs, convolve, normalize_channels = self.bn0_pre, self.conv1_pre, self.bn1_pre
        else:
            inputs.append(local_state)
            normalize_inputs, convolve, normalize_channels = self.bn0, self.conv1, self.bn1
        features = normalize_inputs(torch.stack(inputs, dim=1))
        features = normalize_channels(convolve(self.inp_drop(features)))
        features = self.feature_map_drop(F.relu(features))
        return self.fc(features.flatten(start_dim=1))

    def forward(self, facts, ent_embs, rel_embs, e_r_bias,
                history_tail_seq=None, one_hot_tail_seq=None,
                freq_info=0,
                basic_ent_embs=None, basic_rel_embs=None,
                query_context_state=None, rule_branch_weight=0.0,
                return_components=False, return_auxiliary=False):
        score_list = []
        branch_components = None
        for _ in ent_embs[-1:]:
            ent_emb_i = ent_embs[-1]
            rel_emb_i = rel_embs[-1]
            e_r_bias_i = e_r_bias[-1]

            local_state = None if e_r_bias_i is None else e_r_bias_i[facts[:, 0]]
            x0 = self._project_pair(ent_emb_i[facts[:, 0]], rel_emb_i[facts[:, 1]], local_state)


            x = self.hidden_drop(x0)
            if len(facts) > 1:
                x = self.bn2(x)
            x = F.relu(x)
            if query_context_state is not None:
                query_context_state = query_context_state.to(x.device, dtype=x.dtype)
                query_context_gate = torch.sigmoid(self.query_context_gate(torch.cat([x, query_context_state], dim=1)))
                x = self.query_context_norm(x + query_context_gate * query_context_state)
            score = torch.mm(x, ent_emb_i.transpose(1, 0)).to(self.device)
            rule_weight = float(rule_branch_weight)
            rule_weight = max(0.0, min(rule_weight, 0.95))
            if query_context_state is not None and rule_weight > 0:
                rule_input = torch.cat([x, query_context_state], dim=1)
                rule_score = F.softmax(self.rule_mlp(rule_input), dim=1)
            else:
                rule_weight = 0.0
                rule_score = None

            if freq_info:  # generation mode
                mask_hist = torch.tensor(np.array(one_hot_tail_seq.cpu() == 0, dtype=float)*(-1000),
                                         device=self.device)
                score1 = F.softmax(score + mask_hist, dim=1)

                score2 = self.g_mlp(x)
                score2 = F.softmax(score2, dim=1)

                base_score = score1 * self.alpha + score2 * (1 - self.alpha)
                if rule_weight > 0 and rule_score is not None:
                    score = base_score * (1 - rule_weight) + rule_score * rule_weight
                else:
                    score = base_score
                branch_components = {
                    "repeat": score1 * self.alpha * (1 - rule_weight),
                    "novel": score2 * (1 - self.alpha) * (1 - rule_weight),
                    "rule": rule_score * rule_weight if rule_score is not None else torch.zeros_like(score1),
                }
                score_list.append(score)
            else:
                base_score = F.softmax(score, dim=1)
                if rule_weight > 0 and rule_score is not None:
                    score = base_score * (1 - rule_weight) + rule_score * rule_weight
                else:
                    score = base_score
                score_list.append(score)
                branch_components = {
                    "repeat": torch.zeros_like(score),
                    "novel": base_score * (1 - rule_weight),
                    "rule": rule_score * rule_weight if rule_score is not None else torch.zeros_like(score),
                }

            break


        scores_list = [x.unsqueeze(2) for x in score_list]
        score = torch.cat(scores_list, dim=-1)
        score = torch.sum(score, dim=-1)
        score = torch.log(score)


        auxiliary = None
        if return_auxiliary:
            auxiliary = {
                "historical_logits": self.historical_classifier(x).squeeze(1),
            }

        if return_components and return_auxiliary:
            return score, branch_components, auxiliary
        if return_components:
            return score, branch_components
        if return_auxiliary:
            return score, auxiliary
        return score

    def pretrain_loss(self, facts, facts_neg, E_embs, R_embs, neg_ratio=1):
        E_embs = F.tanh(F.normalize(E_embs))
        e_o = E_embs[facts[:, 2]]
        e_o_neg = E_embs[facts_neg[:, 2]].view(len(facts), neg_ratio, -1)
        x0 = self._project_pair(E_embs[facts[:, 0]], R_embs[facts[:, 1]])
        gap = 6
        positive_score = torch.sum(x0 * e_o, dim=1, keepdim=True)
        negative_score = torch.sum(x0.unsqueeze(1) * e_o_neg, dim=2)
        return F.relu(negative_score + gap - positive_score).mean()

    def pretrain_predict(self, facts, E_embs, R_embs):
        E_embs = F.tanh(F.normalize(E_embs))
        x0 = self._project_pair(E_embs[facts[:, 0]], R_embs[facts[:, 1]])
        scores = torch.mm(x0, E_embs.transpose(1, 0)).to(self.device)
        return scores


class MultiHeadAttention(nn.Module):
    def __init__(self, h_dim, d_k, d_v, n_heads, device):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(h_dim, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(h_dim, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(h_dim, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, h_dim, bias=False)
        self.SDPAtt = ScaledDotProductAttention(d_k)
        self.h_dim = h_dim
        self.d_k = d_k
        self.d_v = d_v  # d_k = d_v = d_q
        self.n_heads = n_heads
        self.device = device

    def forward(self, input_Q, input_K, input_V):  # enc_query_inputs, enc_history_inputs, enc_history_inputs
        '''
        input_Q: (batch_size, 1, h_dim)
        input_K: (batch_size, k, h_dim)
        input_V: (batch_size, k, h_dim)
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # Q: (batch_size, n_heads, 1, d_k)
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,
                                                                                     2)  # K: (batch_size, n_heads, 3, d_k)
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,
                                                                                     2)  # V: (batch_size, n_heads, 3, d_v)

        context = self.SDPAtt(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1,
                                                  self.n_heads * self.d_v)  # context: (batch_size, 1, n_heads * d_v)
        output = self.fc(context)  # (batch_size, 1, h_dim)
        return nn.LayerNorm(self.h_dim).to(self.device)(output + residual)  # (batch_size, 1, h_dim)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q, K, V):
        '''
        Q: (batch_size, n_heads, 1, d_k)
        K: (batch_size, n_heads, 3, d_k)
        V: (batch_size, n_heads, 3, d_v)
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # scores : (batch_size, n_heads, 1, 3)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # (batch_size, n_heads, 1, d_v)
        return context  # (batch_size, n_heads, 1, d_v)


class FeedForwardNet(nn.Module):
    def __init__(self, h_dim, d_ff, device):
        super(FeedForwardNet, self).__init__()
        self.h_dim = h_dim
        self.fc = nn.Sequential(
            nn.Linear(h_dim, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, h_dim, bias=False)
        )
        self.device = device

    def forward(self, inputs):
        '''
        inputs: (batch_size, 1, h_dim)
        '''
        residual = inputs
        output = self.fc(inputs)
        # nn.LayerNorm(x)会对输入的最后一维进行归一化, x需要和输入的最后一维一样大
        return nn.LayerNorm(self.h_dim).to(self.device)(output + residual)  # (batch_size, 1, h_dim_2)


class DecoderLayer(nn.Module):
    def __init__(self, h_dim, d_k, d_v, n_heads, d_ff, device):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(h_dim, d_k, d_v, n_heads, device)
        self.pos_ffn = FeedForwardNet(h_dim, d_ff, device)

    def forward(self, dec_query, dec_K, dec_V):
        '''
        dec_query_inputs: (batch_size, 1, h_dim)
        dec_history_iutputs: (batch_size, k, h_dim) 由dec_inputs得到，长度最长是k=10
        '''
        # enc_query_inputs生成Q矩阵, enc_history_inputs生成K, V矩阵
        dec_outputs = self.dec_self_attn(dec_query, dec_K, dec_V)  # (batch_size, 1, h_dim)
        dec_query_outputs = self.pos_ffn(dec_outputs)  # (batch_size, 1, h_dim)
        return dec_query_outputs  # dec_query_outputs: (batch_size, 1, h_dim) dec_history_outputs: (batch_size, 3, h_dim)
