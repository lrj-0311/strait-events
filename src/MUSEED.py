import torch.nn as nn
import torch
import numpy as np
from layers import *
from decoder import ConvTrans_decoder
from diffusion import ScenarioDiffusionGenerator
import time


class MUSEED(nn.Module):
    """Multi-source uncertainty-aware spatio-temporal event diffusion model.

    The current implementation keeps the stable dynamic encoder and
    copy-generation decoder path runnable, while later MUSE-ED modules can add
    multi-factor evidence encoders and diffusion-based future-state generation.
    """

    def __init__(self, num_E, num_R, h_dim, n_layers, dropout, input_dropout, hidden_dropout, feature_map_dropout,
                 device, alpha, context_dim=0, use_diffusion=False, diffusion_steps=8,
                 diffusion_noise_scale=0.15, diffusion_loss_weight=0.05,
                 use_query_aware_context=False, historical_contrastive_weight=0.0,
                 rule_branch_weight=0.0):
        super(MUSEED, self).__init__()
        self.num_E = num_E
        self.num_R = num_R
        self.device = device
        self.h_dim = h_dim
        self.context_dim = context_dim
        self.use_diffusion = bool(use_diffusion)
        self.diffusion_loss_weight = float(diffusion_loss_weight)
        self.use_query_aware_context = bool(use_query_aware_context and context_dim > 0)
        self.historical_contrastive_weight = float(historical_contrastive_weight)
        self.rule_branch_weight = float(rule_branch_weight)

        self.ent_embs = nn.Parameter(torch.Tensor(num_E, h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.ent_embs)
        self.rel_embs = nn.Parameter(torch.Tensor(num_R * 2, h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.rel_embs)
        self.e_r_bias = nn.Parameter(torch.Tensor(num_E, h_dim), requires_grad=True).float()
        torch.nn.init.zeros_(self.e_r_bias)

        self.global_relation_encoder = DRGlobal_net(num_E, h_dim, device)
        self.local_context_layers = nn.ModuleList(
            [DRlocal_net(h_dim, h_dim, dropout, activation=F.rrelu) for i in range(n_layers)])

        self.structural_evolution_layers = nn.ModuleList(
            [R_GCN(h_dim, h_dim, dropout, activation=F.rrelu) for i in range(n_layers)])

        self.event_decoder = ConvTrans_decoder(num_E, h_dim, device, input_dropout=input_dropout,
                                               hidden_dropout=hidden_dropout,
                                               feature_map_dropout=feature_map_dropout, alpha=alpha)

        if context_dim > 0:
            self.context_encoder = nn.Sequential(
                nn.Linear(context_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(h_dim, h_dim),
            )
            self.context_gate = nn.Linear(h_dim, h_dim)
            if self.use_query_aware_context:
                self.context_feature_embeddings = nn.Parameter(torch.Tensor(context_dim, h_dim))
                torch.nn.init.xavier_normal_(self.context_feature_embeddings)
                self.query_context_projection = nn.Linear(h_dim * 2, h_dim)
                self.query_context_temperature = np.sqrt(h_dim)
            else:
                self.context_feature_embeddings = None
                self.query_context_projection = None
                self.query_context_temperature = np.sqrt(h_dim)
        else:
            self.context_encoder = None
            self.context_gate = None
            self.context_feature_embeddings = None
            self.query_context_projection = None
            self.query_context_temperature = np.sqrt(h_dim)

        if self.use_diffusion:
            diffusion_condition_dim = h_dim if context_dim > 0 else 0
            self.scenario_diffusion = ScenarioDiffusionGenerator(
                h_dim=h_dim,
                num_steps=diffusion_steps,
                noise_scale=diffusion_noise_scale,
                dropout=dropout,
                condition_dim=diffusion_condition_dim,
            )
        else:
            self.scenario_diffusion = None

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        self.loss_nll = torch.nn.NLLLoss()

    def encode_context(self, context_vec):
        if self.context_encoder is None or context_vec is None:
            return None
        if context_vec.dim() == 1:
            context_vec = context_vec.unsqueeze(0)
        context_vec = context_vec.float().to(self.device)
        return self.context_encoder(context_vec).squeeze(0)

    def apply_context(self, ent_embs, context_vec):
        context_state = self.encode_context(context_vec)
        if context_state is None:
            return ent_embs
        context_gate = torch.sigmoid(self.context_gate(context_state)).view(1, -1)
        return F.normalize(ent_embs + context_gate * context_state.view(1, -1), dim=-1)

    def encode_query_context(self, facts, ent_embs, rel_embs, context_vec):
        if (
            not self.use_query_aware_context
            or self.context_feature_embeddings is None
            or self.query_context_projection is None
            or context_vec is None
        ):
            return None, None
        if context_vec.dim() > 1:
            context_vec = context_vec.squeeze(0)
        context_vec = context_vec.float().to(self.device)
        feature_tokens = self.context_feature_embeddings * context_vec.view(-1, 1)
        query_inputs = torch.cat([ent_embs[facts[:, 0]], rel_embs[facts[:, 1]]], dim=1)
        query_state = self.query_context_projection(query_inputs)
        attention_logits = torch.matmul(query_state, feature_tokens.transpose(0, 1)) / self.query_context_temperature
        attention_weights = torch.softmax(attention_logits, dim=1)
        query_context = torch.matmul(attention_weights, feature_tokens)
        return query_context, attention_weights

    def forward(self, hist_glist, use_bias=0, context_vec=None):  # different window size
        ent_embs = F.normalize(self.ent_embs)
        rel_embs = F.normalize(self.rel_embs)
        e_r_bias = F.normalize(self.e_r_bias)
        # rel_embs = self.rel_embs
        # e_r_bias = self.e_r_bias

        ent_embs_list = []
        rel_embs_list = []
        e_r_bias_list = []

        for i, g in enumerate(hist_glist):
            # ---------------------
            tmp_ent_embs = ent_embs
            for layer in self.structural_evolution_layers:  #
                tmp_ent_embs = layer(g, tmp_ent_embs, rel_embs)
                tmp_ent_embs = F.normalize(tmp_ent_embs)
                break
            time_weight = F.sigmoid(torch.mm(ent_embs, self.time_gate_weight) + self.time_gate_bias)
            ent_embs = time_weight * tmp_ent_embs + (1 - time_weight) * ent_embs
            ent_embs = F.normalize(ent_embs)

            # ent_embs = tmp_ent_embs

            # rel_embs = self.global_relation_encoder(g, rel_embs, ent_embs, e_r_bias, self.num_R, i)

            if use_bias:
                for layer in self.local_context_layers:
                    e_r_bias = layer(g, ent_embs, rel_embs, e_r_bias, i)
                    break
            else:
                e_r_bias = None

            ent_embs_list.append(ent_embs)
            rel_embs_list.append(rel_embs)
            e_r_bias_list.append(e_r_bias)
            # e_r_bias_list.append(None)

        if context_vec is not None and ent_embs_list:
            ent_embs_list[-1] = self.apply_context(ent_embs_list[-1], context_vec)

        # return ent_embs, rel_embs, e_r_bias
        return ent_embs_list, rel_embs_list, e_r_bias_list

    def loss_func(self, ghist, facts, facts_neg=None, history_tail_seq=None, one_hot_tail_seq=None, use_bias=0,
                  freq_info=0, neg_ratio=1, context_vec=None, sample_weights=None):
        if ghist is not None:
            ent_embs_list, rel_embs_list, e_r_bias_list = self.forward(ghist, use_bias, context_vec)
            query_context, _ = self.encode_query_context(
                facts, ent_embs_list[-1], rel_embs_list[-1], context_vec
            )
            decoded = self.event_decoder(facts,
                                     ent_embs_list, rel_embs_list, e_r_bias_list,
                                     history_tail_seq, one_hot_tail_seq,
                                     freq_info,
                                     self.ent_embs, self.rel_embs,
                                     query_context_state=query_context,
                                     rule_branch_weight=self.rule_branch_weight,
                                     return_auxiliary=self.historical_contrastive_weight > 0)
            if self.historical_contrastive_weight > 0:
                score, auxiliary = decoded
            else:
                score = decoded
                auxiliary = None
            if sample_weights is None:
                loss = self.loss_nll(score, facts[:, 2])
            else:
                per_query_loss = F.nll_loss(score, facts[:, 2], reduction='none')
                sample_weights = sample_weights.to(score.device, dtype=per_query_loss.dtype)
                loss = (per_query_loss * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
            if (
                self.historical_contrastive_weight > 0
                and auxiliary is not None
                and one_hot_tail_seq is not None
            ):
                historical_logits = auxiliary.get("historical_logits")
                if historical_logits is not None:
                    target_history = one_hot_tail_seq.to(score.device).gather(1, facts[:, 2].view(-1, 1)).squeeze(1)
                    target_history = target_history.float().clamp(0.0, 1.0)
                    historical_loss = F.binary_cross_entropy_with_logits(historical_logits, target_history)
                    loss = loss + self.historical_contrastive_weight * historical_loss
            if self.scenario_diffusion is not None and self.diffusion_loss_weight > 0:
                context_state = self.encode_context(context_vec)
                diffusion_loss = self.scenario_diffusion.reconstruction_loss(ent_embs_list[-1], context_state)
                loss = loss + self.diffusion_loss_weight * diffusion_loss
            return loss
        else:  # 预训练
            loss = self.event_decoder.pretrain_loss(facts, facts_neg,
                                              self.ent_embs, self.rel_embs,
                                              neg_ratio=neg_ratio)
            return loss

    def predict(self, ghist, num_R, facts, history_tail_seq=None, one_hot_tail_seq=None, freq_info=0, use_bias=0,
                context_vec=None, return_uncertainty=False, diffusion_samples=4, use_scenario_mean=False,
                return_components=False, return_query_attention=False):
        with torch.no_grad():
            if ghist is None:
                scores = self.event_decoder.pretrain_predict(facts, self.ent_embs, self.rel_embs)
                return facts, scores

            ent_embs_list, rel_embs_list, e_r_bias_list = self.forward(ghist, use_bias, context_vec)

            query_context, attention_weights = self.encode_query_context(
                facts, ent_embs_list[-1], rel_embs_list[-1], context_vec
            )

            decoded = self.event_decoder(
                facts,
                ent_embs_list,
                rel_embs_list,
                e_r_bias_list,
                history_tail_seq,
                one_hot_tail_seq,
                freq_info,
                self.ent_embs,
                self.rel_embs,
                query_context_state=query_context,
                rule_branch_weight=self.rule_branch_weight,
                return_components=return_components,
            )
            if return_components:
                score, components = decoded
            else:
                score = decoded
                components = None

            if not return_uncertainty or self.scenario_diffusion is None:
                if return_components:
                    if return_query_attention:
                        return facts, score, components, attention_weights
                    return facts, score, components
                return facts, score

            scenario_score, uncertainty = self.decode_scenarios(
                facts,
                ent_embs_list,
                rel_embs_list,
                e_r_bias_list,
                history_tail_seq,
                one_hot_tail_seq,
                freq_info,
                context_vec,
                diffusion_samples,
            )
            if use_scenario_mean:
                score = scenario_score
            if return_components:
                if return_query_attention:
                    return facts, score, uncertainty, components, attention_weights
                return facts, score, uncertainty, components
            return facts, score, uncertainty

    def decode_scenarios(self, facts, ent_embs_list, rel_embs_list, e_r_bias_list,
                         history_tail_seq=None, one_hot_tail_seq=None, freq_info=0,
                         context_vec=None, diffusion_samples=4):
        context_state = self.encode_context(context_vec)
        sampled_ent_states = self.scenario_diffusion.sample(
            ent_embs_list[-1],
            context_state,
            num_samples=diffusion_samples,
        )

        sample_probs = []
        for sampled_ent_state in sampled_ent_states:
            sampled_ent_embs_list = list(ent_embs_list)
            sampled_ent_embs_list[-1] = sampled_ent_state
            sampled_log_score = self.event_decoder(
                facts,
                sampled_ent_embs_list,
                rel_embs_list,
                e_r_bias_list,
                history_tail_seq,
                one_hot_tail_seq,
                freq_info,
                self.ent_embs,
                self.rel_embs,
            )
            sample_probs.append(torch.exp(sampled_log_score))

        sample_probs = torch.stack(sample_probs, dim=0)
        mean_probs = torch.clamp(sample_probs.mean(dim=0), min=1e-12)
        scenario_score = torch.log(mean_probs)
        target_idx = facts[:, 2].view(1, -1, 1).expand(sample_probs.shape[0], -1, 1)
        target_probs = sample_probs.gather(dim=2, index=target_idx).squeeze(2)
        uncertainty = {
            "predictive_variance": sample_probs.var(dim=0, unbiased=False).mean(dim=1),
            "target_std": target_probs.std(dim=0, unbiased=False),
            "entropy": -(mean_probs * torch.log(mean_probs)).sum(dim=1),
            "sample_count": int(sample_probs.shape[0]),
        }
        return scenario_score, uncertainty
