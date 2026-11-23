import torch
import pandas as pd
import os
from tqdm import tqdm
import random
import numpy as np
from config import args
from utils import *
from MUSEED import MUSEED
import dgl

import json
import scipy.sparse as sp
import time

def get_context_vec(mechanism_features, time_idx, device):
    if mechanism_features is None:
        return None
    if time_idx < 0 or time_idx >= mechanism_features.shape[0]:
        return None
    return mechanism_features[time_idx].to(device)


def get_dynamic_model_path(dataset, args):
    suffix = '_coreprotect1' if args.core_protection else ''
    if getattr(args, "checkpoint_tag", ""):
        suffix += '_' + str(args.checkpoint_tag).strip().replace(" ", "_")
    filename = 'model_{}_freqinfo{}_bias{}_alpha{}{}.pth'.format(
        dataset, args.freq_info, args.use_bias, args.alpha, suffix
    )
    return str(RESULTS_ROOT / 'bestmodel' / filename)


def train_eval(args=None, dataset='ACLED_MUSE_7COUNTRY'):
    train_list, valid_list, test_list, train_times, valid_times, test_times, num_ent, num_rel, num_time \
        = dataloader_from_raw(dataset)  # load data
    mechanism_features, mechanism_feature_names = load_mechanism_features(
        dataset, args.mechanism_feature_path, bool(args.mechanism_lag_only),
        selected_columns=args.mechanism_feature_columns,
    )
    core_entity_mask = None
    if args.core_protection:
        group_path = RESULTS_ROOT / 'model_context' / 'entity_group_features.csv'
        if not group_path.exists():
            raise FileNotFoundError('Core protection entity groups not found: {}'.format(group_path))
        group_df = pd.read_csv(group_path).sort_values('id')
        core_entity_mask = group_df['is_core'].to_numpy(dtype=bool)
        print('Core protection enabled: query weight={}'.format(args.core_replay_weight))
    train_g_list, valid_g_list, test_g_list = load_graph(dataset, device)  # load dgl graph, erinfo
    all_facts_filter = train_list+valid_list+test_list
    all_facts_filter = np.concatenate(all_facts_filter)
    all_facts_filter = add_inverse_rel(all_facts_filter, num_rel)  # inverse relations


    if args.freq_info:
        all_times = train_times + valid_times + test_times
        time_tail_freq_list = load_freq(dataset, all_times, num_ent, num_rel)  # history statistics info
    else:
        time_tail_freq_list = None
    print("====================All data loaded!====================")
    # return 0,0,0,0
    # -----------------------------------------load-----------------------------------------

    context_dim = 0 if mechanism_features is None else mechanism_features.shape[1]
    model = MUSEED(num_ent, num_rel, h_dim=args.n_hidden, n_layers=2, dropout=args.dropout,
                   input_dropout=args.input_dropout, hidden_dropout=args.hidden_dropout,
                   feature_map_dropout=args.feat_dropout, device=device, alpha=args.alpha,
                   context_dim=context_dim, use_diffusion=bool(args.use_diffusion),
                   diffusion_steps=args.diffusion_steps,
                   diffusion_noise_scale=args.diffusion_noise_scale,
                   diffusion_loss_weight=args.diffusion_loss_weight,
                   use_query_aware_context=bool(args.use_query_aware_context),
                   historical_contrastive_weight=args.historical_contrastive_weight,
                   rule_branch_weight=args.rule_branch_weight)
    model.to(device)
    if mechanism_feature_names:
        print("Mechanism feature columns:", mechanism_feature_names)
    if args.use_diffusion:
        print("Scenario diffusion enabled: steps={}, samples={}, loss_weight={}".format(
            args.diffusion_steps, args.diffusion_samples, args.diffusion_loss_weight
        ))
    if args.use_query_aware_context:
        print("Query-aware mechanism context enabled.")
    if args.historical_contrastive_weight > 0:
        print("Historical contrastive auxiliary loss enabled: weight={}".format(
            args.historical_contrastive_weight
        ))
    if args.rule_branch_weight > 0:
        print("Learned mechanism-rule branch enabled: weight={}".format(args.rule_branch_weight))

    # ----------------------------------------test----------------------------------------
    if args.mode == 'test':
        print("----------------------------------------test----------------------------------------\n")
        model_state_file = get_dynamic_model_path(dataset, args)
        print("use model file to test: {}".format(model_state_file))
        mrr, hits1, hits3, hits10 = test(model, dataset, train_g_list + valid_g_list, test_g_list, test_list,
                                         time_tail_freq_list, num_ent, num_rel, mode='test', device=device,
                                         model_state_file=model_state_file, all_facts_filter=all_facts_filter,
                                         mechanism_features=mechanism_features,
                                         mechanism_feature_names=mechanism_feature_names)
        print("   mrr in test data:", mrr)
        print(" hits1 in test data:", hits1)
        print(" hits3 in test data:", hits3)
        print("hits10 in test data:", hits10)
        return mrr, hits1, hits3, hits10

    print("----------------------------------------start training----------------------------------------\n")
    best_mrr = 0
    if args.mode == 'pretrain':
        batchsize_dict = {}
        batchsize_dict['YAGO'] = 2048
        batchsize_dict['WIKI'] = 2048
        batchsize_dict['ICEWS14'] = 512
        batchsize_dict['ICEWS18'] = 2048
        batchsize_dict['GDELT'] = 2500

        model_state_file = str(RESULTS_ROOT / 'bestmodel' / 'model_{}_pretrain.pth'.format(dataset))
        os.makedirs(os.path.dirname(model_state_file), exist_ok=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best_mrr = 0
        patience = 0  # early stop
        neg_ratio = 2  # num of neg sample

        batchsize = batchsize_dict.get(dataset, 1024)

        for epoch in range(args.max_epochs):
            model.train()
            losses = []
            train_data = np.concatenate(train_list)
            np.random.shuffle(train_data)
            train_data = add_inverse_rel(train_data, num_rel)  #
            train_data_set = set((item[0], item[1], item[2]) for item in train_data)

            for start_idx in tqdm(range(0, train_data.shape[0], batchsize)):
                end_idx = min(train_data.shape[0], start_idx + batchsize)
                batch_data = train_data[start_idx: end_idx]
                batch_data_neg = []  # 负采样
                for item in batch_data:
                    count = 0
                    while count < neg_ratio:
                        neg_o = np.random.randint(num_ent)
                        if (item[0], item[1], neg_o) not in train_data_set:
                            batch_data_neg.append([item[0], item[1], neg_o])
                            count += 1

                batch_data = torch.tensor(batch_data).long().to(device)
                batch_data_neg = torch.tensor(batch_data_neg).long().to(device)
                loss = model.loss_func(None, batch_data, batch_data_neg, neg_ratio=neg_ratio)
                losses.append(loss.item())
                # print(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print("Epoch {:04d} | Ave Loss: {:.4f}".format(epoch, np.mean(losses)))
            # validate
            if epoch and epoch % args.valid_step == 0:
                ranks = []
                valid_data = np.concatenate(valid_list)
                valid_data = add_inverse_rel(valid_data, num_rel)  #
                for start_idx in range(0, valid_data.shape[0], 512):
                    end_idx = min(valid_data.shape[0], start_idx + 512)
                    batch_data_ = valid_data[start_idx: end_idx]
                    batch_data_ = torch.tensor(batch_data_).long().to(device)
                    facts_output, scores_output = model.predict(None, num_rel, batch_data_)
                    ranks_tmp = get_ranks(facts_output, scores_output, filter=args.filter, num_ent=num_ent)
                    ranks.append(ranks_tmp)
                mrr, hits1, hits3, hits10 = stat_ranks(torch.cat(ranks))

                print("mrr in valid data: {:.4f}".format(mrr))
                if mrr < best_mrr:
                    patience += 1
                    if epoch >= args.max_epochs or patience > 3:  # early stop
                        break
                else:
                    best_mrr = mrr
                    print("!!! Best model update with history_len: {}   Best_mrr:{:.4f}".format(args.train_history_len,
                                                                                                best_mrr))
                    torch.save(
                        {'state_dict': model.state_dict(), 'epoch': epoch, 'history_len': args.train_history_len},
                        model_state_file)  #
        ranks = []
        test_data = np.concatenate(test_list)
        test_data = add_inverse_rel(test_data, num_rel)  #
        for start_idx in range(0, test_data.shape[0], 512):
            end_idx = min(test_data.shape[0], start_idx + 512)
            batch_data_ = test_data[start_idx: end_idx]
            batch_data_ = torch.tensor(batch_data_).long().to(device)
            facts_output, scores_output = model.predict(None, num_rel, batch_data_)
            ranks_tmp = get_ranks(facts_output, scores_output, filter=args.filter, num_ent=num_ent)
            ranks.append(ranks_tmp)
        mrr, hits1, hits3, hits10 = stat_ranks(torch.cat(ranks))
        print("-------Pretrain outcome on {} test data-------".format(dataset))
        print("   mrr in test data:", mrr)
        print(" hits1 in test data:", hits1)
        print(" hits3 in test data:", hits3)
        print("hits10 in test data:", hits10)
        return mrr, hits1, hits3, hits10

    elif args.mode == 'train':
        model_state_file = get_dynamic_model_path(dataset, args)
        os.makedirs(os.path.dirname(model_state_file), exist_ok=True)

        model_state_file_prev = str(RESULTS_ROOT / 'bestmodel' / 'model_{}_pretrain.pth'.format(dataset))
        checkpoint = torch.load(model_state_file_prev, map_location=device)
        model.load_state_dict(checkpoint['state_dict'], strict=False)

        print("Freeze")
        model.ent_embs.requires_grad = False  # Freeze, or the outcome gets worse
        model.rel_embs.requires_grad = False

        # num_params = 0
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         if 'layers' in name:
        #             num_params += param.numel()/2
        #         else:
        #             num_params += param.numel()
        #         print(name)
        # print("model parmameters for {}: {}".format(args.dataset, num_params))
        # return 0, 0, 0, 0



        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2,
                                                               verbose=True,
                                                               threshold=0.01, cooldown=0, min_lr=0.00001, eps=1e-08)
        best_mrr = 0
        patience = 0  # early stop
        for epoch in range(1, args.max_epochs):
            model.train()
            index = [_ for _ in range(len(train_list))]  #
            random.shuffle(index)
            losses = []
            for T_idx in tqdm(index, ncols=100):
                # for T_idx in index:
                if T_idx == 0: continue  #
                # if T_idx - args.train_history_len < 0: continue  #
                output = np.asarray(train_list[T_idx])  #
                output_w_inverse = add_inverse_rel(output, num_rel)
                if args.freq_info:
                    all_tail_seq = time_tail_freq_list[T_idx]  #
                    seq_idx = output_w_inverse[:, 0] * num_rel * 2 + output_w_inverse[:, 1]  #
                    history_tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
                    one_hot_tail_seq = history_tail_seq.masked_fill(history_tail_seq != 0, 1)
                    history_tail_seq.to(device)
                    one_hot_tail_seq.to(device)
                else:
                    history_tail_seq = None
                    one_hot_tail_seq = None
                history_glist = train_g_list[max(0, T_idx - args.train_history_len): T_idx]  #
                sample_weights = None
                if core_entity_mask is not None:
                    core_query = (
                        core_entity_mask[output_w_inverse[:, 0]] |
                        core_entity_mask[output_w_inverse[:, 2]]
                    )
                    sample_weights = torch.tensor(
                        np.where(core_query, args.core_replay_weight, 1.0),
                        dtype=torch.float32,
                        device=device,
                    )
                output_w_inverse = torch.tensor(output_w_inverse).long().to(device)  #
                context_vec = get_context_vec(mechanism_features, T_idx, device)

                loss = model.loss_func(history_glist, output_w_inverse, None, history_tail_seq, one_hot_tail_seq,
                                       args.use_bias, args.freq_info, context_vec=context_vec,
                                       sample_weights=sample_weights)  #
                losses.append(loss.item())
                # print("Timestamp:{}  Loss: {}： ".format(T_idx, loss.item()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # scheduler.step(np.mean(losses))  #
            print("Epoch {:04d} | Ave Loss: {:.4f}".format(epoch, np.mean(losses)))
            # 验证模型效果
            if epoch and epoch % args.valid_step == 0:
                mrr, hits1, hits3, hits10 = test(model, dataset, train_g_list, valid_g_list, valid_list,
                                                 time_tail_freq_list, num_ent, num_rel,
                                                 mode='train', device=device, history_len=args.train_history_len,
                                                 mechanism_features=mechanism_features,
                                                 mechanism_feature_names=mechanism_feature_names,
                                                 export_explanations=False)
                print("mrr in valid data: {:.4f}".format(mrr))
                if mrr < best_mrr:
                    patience += 1
                    if epoch >= args.max_epochs or patience > 3:
                        break
                else:
                    best_mrr = mrr
                    print("!!! Best model update with history_len {}".format(args.train_history_len))
                    torch.save(
                        {'state_dict': model.state_dict(), 'epoch': epoch, 'history_len': args.train_history_len},
                        model_state_file)  # 存储最优模型
        print("Best raw model: {}  best mrr in valid data:{:.4f}".format(model_state_file, best_mrr))
        mrr, hits1, hits3, hits10 = test(model, dataset, train_g_list + valid_g_list, test_g_list, test_list,
                                         time_tail_freq_list, num_ent, num_rel,
                                         mode='test', device=device, model_state_file=model_state_file,
                                         mechanism_features=mechanism_features,
                                         mechanism_feature_names=mechanism_feature_names)
        print("   mrr in test data:", mrr)
        print(" hits1 in test data:", hits1)
        print(" hits3 in test data:", hits3)
        print("hits10 in test data:", hits10)

        return mrr, hits1, hits3, hits10

def test(model, dataset, history_g_list, test_g_list, test_list, time_tail_freq_list, num_ent, num_rel, mode,
         device='cpu', model_state_file=None, history_len=2, all_facts_filter=None, mechanism_features=None,
         mechanism_feature_names=None, export_explanations=True):
    with torch.no_grad():
        if mode == "test":
            checkpoint = torch.load(model_state_file, map_location=device)
            model.load_state_dict(checkpoint['state_dict'], strict=False)  #
            test_history_len = checkpoint['history_len']
            print("Best model from epoch {}, with history length of {}".format(checkpoint['epoch'], test_history_len))
        else:
            test_history_len = history_len

        model.eval()
        ranks = []
        uncertainty_records = []
        diffusion_enabled = bool(args.use_diffusion and getattr(model, "scenario_diffusion", None) is not None)
        explanation_enabled = bool(args.explanation_export_dir) and bool(export_explanations)
        attention_enabled = (
            explanation_enabled
            and bool(args.use_query_aware_context)
            and bool(mechanism_feature_names)
        )
        explanation_records = []
        entity_metric_records = []
        attention_records = []

        for T_idx, test_snap in enumerate(tqdm(test_list, ncols=50)):
            if T_idx - test_history_len < 0:
                history_glist = [g for g in history_g_list[T_idx - test_history_len:]] \
                                + [g for g in test_g_list[0: T_idx]]
            else:
                history_glist = [g for g in test_g_list[T_idx - test_history_len: T_idx]]
            output_w_inverse = add_inverse_rel(test_snap, num_rel)
            output_w_inverse = torch.LongTensor(output_w_inverse)
            global_time_idx = len(history_g_list) + T_idx
            context_vec = get_context_vec(mechanism_features, global_time_idx, device)
            if args.freq_info:
                all_tail_seq = time_tail_freq_list[len(history_g_list) + T_idx]
                seq_idx = output_w_inverse[:, 0] * num_rel * 2 + output_w_inverse[:, 1]  # (h,r ,?)
                history_tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())  #
                one_hot_tail_seq = history_tail_seq.masked_fill(history_tail_seq != 0, 1)
                # history_tail_seq.to(device)
                # one_hot_tail_seq.to(device)
                # all_static_tail_seq = time_tail_freq_list[-1]
                # one_hot_tail_seq = history_tail_seq.masked_fill(torch.tensor(all_static_tail_seq[seq_idx].todense()) != 0, 1)
            else:
                history_tail_seq = None
                one_hot_tail_seq = None

            if diffusion_enabled:
                prediction = model.predict(
                    history_glist, num_rel, output_w_inverse.to(device),
                    history_tail_seq,
                    one_hot_tail_seq,
                    args.freq_info, args.use_bias,
                    context_vec=context_vec,
                    return_uncertainty=True,
                    diffusion_samples=args.diffusion_samples,
                    use_scenario_mean=bool(args.diffusion_use_mean_score),
                    return_components=explanation_enabled,
                    return_query_attention=attention_enabled,
                )
                if explanation_enabled:
                    if attention_enabled:
                        facts_output, scores_output, uncertainty, branch_components, query_attention = prediction
                    else:
                        facts_output, scores_output, uncertainty, branch_components = prediction
                        query_attention = None
                else:
                    facts_output, scores_output, uncertainty = prediction
                    branch_components = None
                    query_attention = None
                if uncertainty is not None:
                    predictive_variance = uncertainty["predictive_variance"].detach().cpu()
                    target_std = uncertainty["target_std"].detach().cpu()
                    entropy = uncertainty["entropy"].detach().cpu()
                    uncertainty_records.append({
                        "mode": mode,
                        "local_time_idx": T_idx,
                        "global_time_idx": global_time_idx,
                        "num_facts": int(output_w_inverse.shape[0]),
                        "diffusion_samples": int(uncertainty["sample_count"]),
                        "mean_predictive_variance": float(predictive_variance.mean().item()),
                        "max_predictive_variance": float(predictive_variance.max().item()),
                        "mean_target_std": float(target_std.mean().item()),
                        "max_target_std": float(target_std.max().item()),
                        "mean_entropy": float(entropy.mean().item()),
                    })
            else:
                prediction = model.predict(
                    history_glist, num_rel, output_w_inverse.to(device),
                    history_tail_seq,
                    one_hot_tail_seq,
                    args.freq_info, args.use_bias,
                    context_vec=context_vec,
                    return_components=explanation_enabled,
                    return_query_attention=attention_enabled,
                )
                if explanation_enabled:
                    if attention_enabled:
                        facts_output, scores_output, branch_components, query_attention = prediction
                    else:
                        facts_output, scores_output, branch_components = prediction
                        query_attention = None
                else:
                    facts_output, scores_output = prediction
                    branch_components = None
                    query_attention = None
            ranks_tmp = get_ranks(facts_output, scores_output, filter=args.filter, num_ent=num_ent, all_facts_filter=all_facts_filter)
            ranks.append(ranks_tmp)
            if explanation_enabled:
                facts_cpu = facts_output.detach().cpu()
                scores_cpu = scores_output.detach().cpu()
                probabilities = torch.exp(scores_cpu)
                top_values, top_indices = torch.topk(probabilities, k=min(2, probabilities.shape[1]), dim=1)
                margins = top_values[:, 0] if top_values.shape[1] == 1 else top_values[:, 0] - top_values[:, 1]
                selected = torch.topk(
                    margins,
                    k=min(int(args.explanation_top_k), len(margins)),
                ).indices
                component_cpu = {
                    name: value.detach().cpu() for name, value in (branch_components or {}).items()
                }
                attention_cpu = query_attention.detach().cpu() if query_attention is not None else None
                for idx in selected.tolist():
                    predicted_tail = int(top_indices[idx, 0].item())
                    contribution = {
                        name: float(value[idx, predicted_tail].item())
                        for name, value in component_cpu.items()
                    }
                    contribution_total = sum(contribution.values())
                    if contribution_total > 0:
                        contribution = {
                            name: value / contribution_total for name, value in contribution.items()
                        }
                    explanation_records.append({
                        "local_time_idx": T_idx,
                        "global_time_idx": global_time_idx,
                        "head_id": int(facts_cpu[idx, 0].item()),
                        "relation_id": int(facts_cpu[idx, 1].item()),
                        "true_tail_id": int(facts_cpu[idx, 2].item()),
                        "predicted_tail_id": predicted_tail,
                        "predicted_probability": float(top_values[idx, 0].item()),
                        "score_margin": float(margins[idx].item()),
                        "true_rank": float(ranks_tmp[idx].item()),
                        "repeat_contribution": contribution.get("repeat", 0.0),
                        "novel_contribution": contribution.get("novel", 0.0),
                        "rule_contribution": contribution.get("rule", 0.0),
                    })
                for idx in range(len(facts_cpu)):
                    predicted_tail = int(top_indices[idx, 0].item())
                    predicted_probability = float(top_values[idx, 0].item())
                    true_rank = float(ranks_tmp[idx].item())
                    entity_metric_records.append({
                        "global_time_idx": global_time_idx,
                        "head_id": int(facts_cpu[idx, 0].item()),
                        "relation_id": int(facts_cpu[idx, 1].item()),
                        "true_tail_id": int(facts_cpu[idx, 2].item()),
                        "predicted_tail_id": predicted_tail,
                        "predicted_probability": predicted_probability,
                        "true_rank": true_rank,
                    })
                    if attention_cpu is not None:
                        att_values = attention_cpu[idx]
                        top_feature_idx = int(torch.argmax(att_values).item())
                        attention_record = {
                            "global_time_idx": global_time_idx,
                            "local_time_idx": T_idx,
                            "head_id": int(facts_cpu[idx, 0].item()),
                            "relation_id": int(facts_cpu[idx, 1].item()),
                            "true_tail_id": int(facts_cpu[idx, 2].item()),
                            "predicted_tail_id": predicted_tail,
                            "predicted_probability": predicted_probability,
                            "true_rank": true_rank,
                            "top_attention_feature": mechanism_feature_names[top_feature_idx],
                            "top_attention_weight": float(att_values[top_feature_idx].item()),
                        }
                        for feature_idx, feature_name in enumerate(mechanism_feature_names):
                            attention_record["attention_{}".format(feature_name)] = float(
                                att_values[feature_idx].item()
                            )
                        attention_records.append(attention_record)
        mrr, hits1, hits3, hits10 = stat_ranks(torch.cat(ranks))
        if uncertainty_records:
            uncertainty_df = pd.DataFrame(uncertainty_records)
            print("Diffusion uncertainty summary:")
            print("  mean predictive variance: {:.6f}".format(uncertainty_df["mean_predictive_variance"].mean()))
            print("  mean target std: {:.6f}".format(uncertainty_df["mean_target_std"].mean()))
            print("  mean entropy: {:.6f}".format(uncertainty_df["mean_entropy"].mean()))
            if args.diffusion_export_path:
                export_dir = os.path.dirname(args.diffusion_export_path)
                if export_dir:
                    os.makedirs(export_dir, exist_ok=True)
                uncertainty_df.to_csv(args.diffusion_export_path, index=False)
                print("  exported:", args.diffusion_export_path)
        if explanation_records:
            os.makedirs(args.explanation_export_dir, exist_ok=True)
            entity_map = pd.read_csv(DATA_ROOT / dataset / 'entity2id.csv').sort_values('id')
            relation_map = pd.read_csv(DATA_ROOT / dataset / 'relation2id.csv').sort_values('id')
            time_map = pd.read_csv(DATA_ROOT / dataset / 'time_slice_features.csv')
            entity_names = entity_map['name'].tolist()
            relation_names = relation_map['name'].tolist()
            timestamp_by_id = time_map.set_index('timestamp_id')['timestamp_start'].to_dict()

            explanation_df = pd.DataFrame(explanation_records)
            explanation_df['timestamp_start'] = explanation_df['global_time_idx'].map(timestamp_by_id)
            explanation_df['head'] = explanation_df['head_id'].map(lambda idx: entity_names[idx])
            explanation_df['true_tail'] = explanation_df['true_tail_id'].map(lambda idx: entity_names[idx])
            explanation_df['predicted_tail'] = explanation_df['predicted_tail_id'].map(lambda idx: entity_names[idx])
            explanation_df['relation'] = explanation_df['relation_id'].map(
                lambda idx: relation_names[idx] if idx < num_rel else 'INVERSE::' + relation_names[idx - num_rel]
            )
            explanation_df.to_csv(
                os.path.join(args.explanation_export_dir, 'museed_top_predicted_facts.csv'),
                index=False,
            )
            explanation_df.groupby(['timestamp_start'])[
                ['repeat_contribution', 'novel_contribution', 'rule_contribution']
            ].mean().reset_index().to_csv(
                os.path.join(args.explanation_export_dir, 'museed_branch_contributions.csv'),
                index=False,
            )
            explanation_df.groupby(['head', 'predicted_tail'], as_index=False).agg(
                predicted_frequency=('predicted_tail_id', 'size'),
                mean_probability=('predicted_probability', 'mean'),
            ).sort_values('predicted_frequency', ascending=False).to_csv(
                os.path.join(args.explanation_export_dir, 'museed_predicted_entity_edge_frequency.csv'),
                index=False,
            )

            metric_df = pd.DataFrame(entity_metric_records)
            metric_df['head'] = metric_df['head_id'].map(lambda idx: entity_names[idx])
            metric_df['true_tail'] = metric_df['true_tail_id'].map(lambda idx: entity_names[idx])
            metric_df['predicted_tail'] = metric_df['predicted_tail_id'].map(lambda idx: entity_names[idx])
            metric_df['timestamp_start'] = metric_df['global_time_idx'].map(timestamp_by_id)
            metric_df['relation'] = metric_df['relation_id'].map(
                lambda idx: relation_names[idx] if idx < num_rel else 'INVERSE::' + relation_names[idx - num_rel]
            )
            combined_names = metric_df['head'].astype(str) + '|' + metric_df['true_tail'].astype(str)
            metric_df['is_china'] = combined_names.str.contains('China', case=False, regex=False)
            metric_df['is_taiwan'] = combined_names.str.contains('Taiwan', case=False, regex=False)
            metric_df['is_core'] = metric_df['is_china'] | metric_df['is_taiwan']
            metric_df.to_csv(
                os.path.join(args.explanation_export_dir, 'museed_all_query_predictions.csv'),
                index=False,
            )
            metric_rows = []
            for group_name, mask in [
                ('overall', np.ones(len(metric_df), dtype=bool)),
                ('core', metric_df['is_core']),
                ('china', metric_df['is_china']),
                ('taiwan', metric_df['is_taiwan']),
            ]:
                group_ranks = metric_df.loc[mask, 'true_rank'].to_numpy(dtype=float)
                metric_rows.append({
                    'entity_group': group_name,
                    'num_queries': len(group_ranks),
                    'mrr': float(np.mean(1.0 / group_ranks)),
                    'hits_1': float(np.mean(group_ranks <= 1)),
                    'hits_3': float(np.mean(group_ranks <= 3)),
                    'hits_10': float(np.mean(group_ranks <= 10)),
                })
            pd.DataFrame(metric_rows).to_csv(
                os.path.join(args.explanation_export_dir, 'museed_per_entity_metrics.csv'),
                index=False,
            )
            if attention_records:
                attention_df = pd.DataFrame(attention_records)
                attention_df['timestamp_start'] = attention_df['global_time_idx'].map(timestamp_by_id)
                attention_df['head'] = attention_df['head_id'].map(lambda idx: entity_names[idx])
                attention_df['true_tail'] = attention_df['true_tail_id'].map(lambda idx: entity_names[idx])
                attention_df['predicted_tail'] = attention_df['predicted_tail_id'].map(lambda idx: entity_names[idx])
                attention_df['relation'] = attention_df['relation_id'].map(
                    lambda idx: relation_names[idx] if idx < num_rel else 'INVERSE::' + relation_names[idx - num_rel]
                )
                combined_names = attention_df['head'].astype(str) + '|' + attention_df['true_tail'].astype(str)
                attention_df['is_china'] = combined_names.str.contains('China', case=False, regex=False)
                attention_df['is_taiwan'] = combined_names.str.contains('Taiwan', case=False, regex=False)
                attention_df['is_core'] = attention_df['is_china'] | attention_df['is_taiwan']
                attention_df.to_csv(
                    os.path.join(args.explanation_export_dir, 'museed_query_attention_weights.csv'),
                    index=False,
                )

                attention_cols = [
                    "attention_{}".format(feature_name)
                    for feature_name in mechanism_feature_names
                    if "attention_{}".format(feature_name) in attention_df.columns
                ]
                summary_rows = []
                for group_name, mask in [
                    ('overall', np.ones(len(attention_df), dtype=bool)),
                    ('core', attention_df['is_core']),
                    ('china', attention_df['is_china']),
                    ('taiwan', attention_df['is_taiwan']),
                ]:
                    group_attention = attention_df.loc[mask, attention_cols]
                    if group_attention.empty:
                        continue
                    means = group_attention.mean()
                    top_col = means.sort_values(ascending=False).index[0]
                    row = {
                        'entity_group': group_name,
                        'num_queries': int(len(group_attention)),
                        'top_attention_feature': top_col.replace('attention_', '', 1),
                        'top_attention_weight': float(means[top_col]),
                    }
                    for col in attention_cols:
                        row[col] = float(means[col])
                    summary_rows.append(row)
                pd.DataFrame(summary_rows).to_csv(
                    os.path.join(args.explanation_export_dir, 'museed_query_attention_summary.csv'),
                    index=False,
                )
                attention_df.groupby('timestamp_start', as_index=False)[attention_cols].mean().to_csv(
                    os.path.join(args.explanation_export_dir, 'museed_weekly_query_attention_summary.csv'),
                    index=False,
                )
            print("Neural explanation outputs exported:", args.explanation_export_dir)
    return mrr, hits1, hits3, hits10


if __name__ == '__main__':
    seed = int(args.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    device = torch.device('cuda:{}'.format(args.device_id) if torch.cuda.is_available() else 'cpu')
    dataset = args.dataset

    print(args)

    mrr_list = []
    hits1_list = []
    hits3_list = []
    hits10_list = []

    for round in range(1):
        mrr, hits1, hits3, hits10 = train_eval(args, dataset)  #

        mrr_list.append(mrr)
        hits1_list.append(hits1)
        hits3_list.append(hits3)
        hits10_list.append(hits10)

    print(mrr_list)
    print(hits1_list)
    print(hits3_list)
    print(hits10_list)

    mrr_mean = np.mean(mrr_list)
    mrr_var = np.var(mrr_list)
    mrr_mean = ('%.4f' % mrr_mean)
    mrr_var = ('%.4f' % mrr_var)

    hits1_mean = np.mean(hits1_list)
    hits1_var = np.var(hits1_list)
    hits1_mean = ('%.4f' % hits1_mean)
    hits1_var = ('%.4f' % hits1_var)

    hits3_mean = np.mean(hits3_list)
    hits3_var = np.var(hits3_list)
    hits3_mean = ('%.4f' % hits3_mean)
    hits3_var = ('%.4f' % hits3_var)

    hits10_mean = np.mean(hits10_list)
    hits10_var = np.var(hits10_list)
    hits10_mean = ('%.4f' % hits10_mean)
    hits10_var = ('%.4f' % hits10_var)

    res_path = '../eva_result_final.csv'
    exp_result = []
    model_name = 'model_{}_freqinfo{}_verse={}_alpha={}.pth'.format(dataset, args.freq_info, args.use_bias, args.alpha)
    heads = ['dataset', 'dim', 'lr', 'freq_info', 'use_bias', 'alpha', 'mrr', 'hits1', 'hits3', 'hits10', 'mode',
             'history_len']
    exp_result.append(
        (dataset, args.n_hidden, args.lr, args.freq_info, args.use_bias, args.alpha,
         "{}+-{}".format(mrr_mean, mrr_var),
         "{}+-{}".format(hits1_mean, hits1_var),
         "{}+-{}".format(hits3_mean, hits3_var),
         "{}+-{}".format(hits10_mean, hits10_var),
         args.mode, args.train_history_len))

    # exp_result.append(
    #     (dataset, args.n_hidden, args.lr, args.freq_info, args.use_bias, args.alpha,
    #      "{}".format(mrr_mean),
    #      "{}".format(hits1_mean),
    #      "{}".format(hits3_mean),
    #      "{}".format(hits10_mean),
    #      'non-freeze', args.train_history_len))

    # exp_result = pd.DataFrame(exp_result, columns=heads)
    # if os.path.exists(res_path):
    #     exp_result.to_csv(res_path, header=0, index=False, mode='a')
    # else:
    #     exp_result.to_csv(res_path, index=False)
