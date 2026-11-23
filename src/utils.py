import collections
import random
import time

import torch
import torch.nn as nn
import dgl
import numpy as np
from collections import defaultdict
import pandas as pd
import json
import os
import scipy.sparse as sp
from tqdm import tqdm
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = WORKSPACE_ROOT / "data"
CACHE_ROOT = WORKSPACE_ROOT / "data_cached"
RESULTS_ROOT = WORKSPACE_ROOT / "results"


def canonical_dataset_name(dataset):
    if '_delete' in dataset:
        return dataset.split('_delete')[0]
    return dataset


def build_sub_graph(num_nodes, num_rels, num_types, facts, device):
    """Construct a directed snapshot with inverse relations and degree weights."""
    facts = np.asarray(facts)
    if facts.ndim != 2 or facts.shape[1] < 3:
        raise ValueError("facts must have subject, relation, and object columns")

    subject = torch.as_tensor(facts[:, 0], dtype=torch.long)
    relation = torch.as_tensor(facts[:, 1], dtype=torch.long)
    obj = torch.as_tensor(facts[:, 2], dtype=torch.long)
    source = torch.cat((subject, obj))
    target = torch.cat((obj, subject))
    edge_type = torch.cat((relation, relation + num_rels))

    graph = dgl.graph((source, target), num_nodes=num_nodes)
    in_norm = graph.in_degrees().clamp_min(1).float().reciprocal()
    out_norm = graph.out_degrees().clamp_min(1).float().reciprocal()
    graph.ndata['id'] = torch.arange(num_nodes, dtype=torch.long).unsqueeze(1)
    graph.ndata['in_norm'] = in_norm.unsqueeze(1)
    graph.ndata['out_norm'] = out_norm.unsqueeze(1)
    graph.edata['type'] = edge_type
    graph.edata['up_norm'] = in_norm.index_select(0, source).unsqueeze(1)
    return graph.to(device)


def e_r_info(facts, num_rels):  # triplets: [[s, r, o], [s, r, o], ...]
    # inverse_facts = facts[:, [2, 1, 0, 3, 5, 4]]
    inverse_facts = facts[:, [2, 1, 0, 3]]
    inverse_facts[:, 1] = inverse_facts[:, 1] + num_rels
    facts = np.concatenate([facts, inverse_facts])
    # src, rel, dst, _, src_type, dst_type = facts.transpose()

    src, rel, dst, _ = facts.transpose()  #  (sub, rel, obj, timestamp, sub_type, obj_type)
    uniq_e = np.unique(src)
    uniq_r = np.unique(rel)
    e_to_r = defaultdict(set)
    r_to_r = defaultdict(set)
    r_to_e = defaultdict(set)
    df_now = pd.DataFrame(facts)
    for idx in range(df_now.shape[0]):
        item = df_now.iloc[idx]
        h = item[0]
        r = item[1]
        t = item[2]
        co_occur_r = set(df_now[(df_now[0] == h) & (df_now[1] != r)][1].tolist())
        for r_ in co_occur_r:
            r_to_r[r].add(r_)
        e_to_r[h].add(r)
        r_to_e[r].add(h)
        r_to_e[r].add(t)

    r_to_r_list = []
    for r in uniq_r:
        r_to_r[r].add(int(r))
        tmp_list = np.array(list(r_to_r[r])).tolist()
        r_to_r_list.append(tmp_list)

    r_to_e_list = []
    for r in uniq_r:
        tmp_list = np.array(list(r_to_e[r])).tolist()
        # print(type(tmp_list[0]))
        # tmp_list = np.array(r_to_e[r]).tolist()
        r_to_e_list.append(tmp_list)
    info_dict = {}
    # info_dict['uniq_e'] = uniq_e.tolist()
    # info_dict['e_to_r'] = e_to_r_list
    info_dict['uniq_r'] = uniq_r.tolist()
    info_dict['r_to_r'] = r_to_r_list
    info_dict['r_to_e'] = r_to_e_list

    return info_dict


def get_ranks(test_facts, scores, batch_size=1000, filter=0, num_ent=0, all_facts_filter=None):
    ranks = []
    if filter:
        all_facts = test_facts.to('cpu').numpy()
        all_facts_set = set((item[0], item[1], item[2]) for item in all_facts)

        if all_facts_filter is not None:
            # all_facts_filter = np.array(all_facts_filter)
            all_facts_set = set((item[0], item[1], item[2]) for item in all_facts_filter)
            # print("filtered all")

        for idx in range(len(all_facts)):
            tmp_fact = all_facts[idx]
            score = scores[idx]
            filter_t = []
            # print((tmp_fact[0], tmp_fact[1], tmp_fact[2]) in all_facts_set)
            for t_ in range(num_ent):
                if (tmp_fact[0], tmp_fact[1], t_) in all_facts_set:
                    filter_t.append(t_)
            filter_t.remove(tmp_fact[2])
            score[np.asarray(filter_t)] = -np.inf
            _, indices = torch.sort(score, descending=True)
            indice = torch.nonzero(indices == tmp_fact[2]).item()
            ranks.append(indice)
        ranks = torch.tensor(ranks)
        ranks += 1
    else:
        num_facts = len(test_facts)
        n_batch = (num_facts + batch_size - 1) // batch_size
        ranks = []
        for idx in range(n_batch):
            batch_start = idx * batch_size
            batch_end = min(num_facts, (idx + 1) * batch_size)
            batch_facts = test_facts[batch_start:batch_end, :]
            batch_scores = scores[batch_start:batch_end, :]
            target = test_facts[batch_start:batch_end, 2]
            ranks.append(sort_rank(batch_scores, target))
        ranks = torch.cat(ranks)
        ranks += 1
    return ranks


def sort_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def stat_ranks(ranks):
    mrr = torch.mean(1.0 / ranks.float())
    hits1 = torch.mean((ranks <= 1).float())
    hits3 = torch.mean((ranks <= 3).float())
    hits10 = torch.mean((ranks <= 10).float())
    return mrr.item(), hits1.item(), hits3.item(), hits10.item()


def add_inverse_rel(data, num_rel):  # [[s, r, o], [], ...]
    inverse_triples = np.array([[o, r + num_rel, s, t] for s, r, o, t in data])
    triples_w_inverse = np.concatenate((data, inverse_triples))
    return triples_w_inverse


#####################################Load data#####################################
def load_quadruples(in_Path, file_name):
    with open(os.path.join(in_Path, file_name), 'r') as f_read:
        quadruple_list = []
        timestamp_all = set()
        for line in f_read:
            line_splitted = line.split()
            head = int(line_splitted[0])
            relation = int(line_splitted[1])
            tail = int(line_splitted[2])
            timestamp = int(line_splitted[3])
            # quadruple_list.append((h, r, t, timestamp))
            quadruple_list.append((head, relation, tail, timestamp))
            timestamp_all.add(timestamp)
    timestamp_all = list(timestamp_all)
    timestamp_all.sort()
    return np.asarray(quadruple_list), timestamp_all


def dataloader_from_raw(dataset):
    print("==============load dataloader from raw files")

    dataset = canonical_dataset_name(dataset)

    data_path = str(DATA_ROOT / dataset)
    # data_path = 'data/{}'.format(dataset)
    train_data, train_times = load_quadruples(data_path, 'train.txt')
    valid_data, valid_times = load_quadruples(data_path, 'valid.txt')
    test_data, test_times = load_quadruples(data_path, 'test.txt')
    data_all = np.concatenate([train_data, valid_data, test_data])

    df_all = pd.DataFrame(data_all)
    num_ent = max(set(df_all[0].tolist() + df_all[2].tolist())) + 1
    num_rel = max(set(df_all[1].tolist())) + 1
    num_time = len(set(df_all[3].tolist())) + 1

    train_list = []
    valid_list = []
    test_list = []

    for T in train_times:
        facts = df_all[df_all[3] == T].values.tolist()
        train_list.append(facts)
    for T in valid_times:
        facts = df_all[df_all[3] == T].values.tolist()
        valid_list.append(facts)
    for T in test_times:
        facts = df_all[df_all[3] == T].values.tolist()
        test_list.append(facts)

    # for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    #     ratio = 0.2
    #     train_list = []
    #     valid_list = []
    #     test_list = []
    #     num_ent_delete = random.sample(range(0,num_ent), int(num_ent*ratio))
    #     print("left ents: ",len(num_ent_delete))
    #     all_trainsamples = 0
    #     for T in train_times:
    #         facts = df_all[(df_all[3] == T) & (df_all[0].isin(num_ent_delete))& (df_all[2].isin(num_ent_delete))].values.tolist()
    #         train_list.append(facts)
    #         all_trainsamples +=len(facts)
    #     print("left train samples: ", all_trainsamples)
    #     # if(all_trainsamples/len(num_ent_delete)>30):
    #     if(1):
    #         print("find: ", len(num_ent_delete))
    #         for T in valid_times:
    #             facts = df_all[(df_all[3] == T) & (df_all[0].isin(num_ent_delete))& (df_all[2].isin(num_ent_delete))].values.tolist()
    #             valid_list.append(facts)
    #         for T in test_times:
    #             facts = df_all[(df_all[3] == T) & (df_all[0].isin(num_ent_delete))& (df_all[2].isin(num_ent_delete))].values.tolist()
    #             test_list.append(facts)
    #         break
    #     else: continue
    return train_list, valid_list, test_list, train_times, valid_times, test_times, num_ent, num_rel, num_time


def load_mechanism_features(dataset, feature_path, lag_only=True, selected_columns=None):
    if not feature_path:
        return None, []
    if not os.path.exists(feature_path):
        print("Mechanism feature file not found, skip:", feature_path)
        return None, []

    dataset = canonical_dataset_name(dataset)
    time_path = str(DATA_ROOT / dataset / 'time_slice_features.csv')
    if not os.path.exists(time_path):
        print("Dataset time feature file not found, skip mechanism features:", time_path)
        return None, []

    time_df = pd.read_csv(time_path)
    feature_df = pd.read_csv(feature_path)
    if 'timestamp_start' not in time_df.columns or 'timestamp_start' not in feature_df.columns:
        print("timestamp_start is required for mechanism feature alignment.")
        return None, []

    if lag_only:
        print("Mechanism feature mode: prediction-safe lag-only features.")
    else:
        print("Mechanism feature mode: synchronous analysis features enabled; check leakage before prediction.")

    # The dataset time map is authoritative; avoid suffixing duplicate metadata
    # columns that may also be present in exported mechanism feature files.
    feature_df = feature_df.drop(columns=['timestamp_id', 'split'], errors='ignore')
    merged = time_df[['timestamp_id', 'timestamp_start', 'split']].merge(
        feature_df, on='timestamp_start', how='left'
    ).sort_values('timestamp_id')

    blocked_prefixes = ('risk_t_plus', 'risk_count_t_plus', 'target_', 'label_')
    blocked_columns = {'timestamp_id', 'timestamp_start', 'split'}
    requested = set(selected_columns or [])
    numeric_columns = []
    for column in merged.columns:
        if column in blocked_columns or column.startswith(blocked_prefixes):
            continue
        if lag_only and '_lag_' not in column:
            continue
        if requested and column not in requested:
            continue
        if pd.api.types.is_numeric_dtype(merged[column]):
            numeric_columns.append(column)

    if requested:
        missing = sorted(requested - set(numeric_columns))
        if missing:
            print("Requested mechanism columns not used:", missing)

    if not numeric_columns:
        print("No usable mechanism feature columns found.")
        return None, []

    values = merged[numeric_columns].fillna(0.0).astype('float32')
    train_values = values.loc[merged['split'].eq('train')]
    if train_values.empty:
        raise ValueError("Mechanism feature standardization requires training rows.")
    train_mean = train_values.mean(axis=0)
    train_std = train_values.std(axis=0).replace(0, 1.0)
    values = (values - train_mean) / train_std
    feature_tensor = torch.tensor(values.to_numpy(), dtype=torch.float32)
    print("Loaded mechanism features: {} rows, {} columns".format(feature_tensor.shape[0], feature_tensor.shape[1]))
    return feature_tensor, numeric_columns


def load_freq(dataset, all_times, num_ent, num_rel):
    print("==============load freq info")
    freq_path = str(CACHE_ROOT)
    if not os.path.exists(freq_path):
        os.makedirs(freq_path)
    time_tail_seq_list = []
    time_tail_freq_list = []
    timelast = time.time()
    for idx in tqdm(range(len(all_times)), ncols=50):
        time_tail_seq = sp.load_npz(str(
            CACHE_ROOT / dataset / 'history_seq' / 'h_r_history_seq_{}.npz'.format(idx)
        ))
        time_tail_seq_list.append(time_tail_seq)

    for idx in tqdm(range(len(all_times)), ncols=50):
        if idx == 0:
            time_tail_freq = sp.csr_matrix(([], ([], [])), shape=(num_ent * (num_rel * 2), num_ent))
            time_tail_freq_list.append(time_tail_freq)
            continue
        else:
            time_tail_freq = time_tail_seq_list[idx - 1] + time_tail_freq_list[idx - 1]
            time_tail_freq_list.append(time_tail_freq)
        #
        # if len(time_tail_freq_list)<4:
        #     for item in time_tail_seq_list[:-1]:
        #         time_tail_freq += item
        # else:
        #     for item in time_tail_seq_list[-4:-1]:
        #         time_tail_freq +=item
        #
        # time_tail_freq+=time_tail_freq_list[-1]
    return time_tail_freq_list


def load_graph(dataset, device):
    print("==============load graphs and json files")
    dgl_path = str(CACHE_ROOT / dataset / 'dgl_graph' / '{}_{}')
    json_path = str(CACHE_ROOT / dataset / 'e_r_info_json' / '{}_{}')
    if (os.path.exists(dgl_path.format(dataset, 'train.dgl'))):
        train_g_list, _ = dgl.load_graphs(dgl_path.format(dataset, 'train.dgl'))
        valid_g_list, _ = dgl.load_graphs(dgl_path.format(dataset, 'valid.dgl'))
        test_g_list, _ = dgl.load_graphs(dgl_path.format(dataset, 'test.dgl'))
    else:
        print("===============graphs and e_r_info files not exist, process...")
        if not os.path.exists(str(CACHE_ROOT / dataset / 'dgl_graph')):
            os.makedirs(str(CACHE_ROOT / dataset / 'dgl_graph'))
            os.makedirs(str(CACHE_ROOT / dataset / 'e_r_info_json'))
            print("build new dir!")
        train_list, valid_list, test_list, train_times, valid_times, test_times, num_ent, num_rel, num_time = dataloader_from_raw(
            dataset)
        train_g_list = [build_sub_graph(num_ent, num_rel, None, np.asarray(snap), device) for snap in train_list]
        valid_g_list = [build_sub_graph(num_ent, num_rel, None, np.asarray(snap), device) for snap in valid_list]
        test_g_list = [build_sub_graph(num_ent, num_rel, None, np.asarray(snap), device) for snap in test_list]
        dgl.save_graphs(dgl_path.format(dataset, 'train.dgl'), train_g_list)
        dgl.save_graphs(dgl_path.format(dataset, 'valid.dgl'), valid_g_list)
        dgl.save_graphs(dgl_path.format(dataset, 'test.dgl'), test_g_list)
        train_info_dict = {}
        for i, snap in enumerate(train_list):
            train_info_dict[i] = e_r_info(np.asarray(snap), num_rel)
        with open(json_path.format(dataset, 'train.json'), 'w') as f:
            json.dump(train_info_dict, f)
        valid_info_dict = {}
        for i, snap in enumerate(valid_list):
            valid_info_dict[i] = e_r_info(np.asarray(snap), num_rel)
        with open(json_path.format(dataset, 'valid.json'), 'w') as f:
            json.dump(valid_info_dict, f)
        test_info_dict = {}
        for i, snap in enumerate(test_list):
            test_info_dict[i] = e_r_info(np.asarray(snap), num_rel)
        with open(json_path.format(dataset, 'test.json'), 'w') as f:
            json.dump(test_info_dict, f)
        print("===============graphs and e_r_info files: Done!")

    train_g_list = [g.to(device) for g in train_g_list]
    valid_g_list = [g.to(device) for g in valid_g_list]
    test_g_list = [g.to(device) for g in test_g_list]

    with open(json_path.format(dataset, 'train.json'), 'r') as f:
        info_dict = json.load(f)
        for i in range(len(train_g_list)):
            item = info_dict['{}'.format(i)]
            # train_g_list[i].uniq_e = item['uniq_e']
            train_g_list[i].uniq_r = item['uniq_r']
            # train_g_list[i].e_to_r = item['e_to_r']
            train_g_list[i].r_to_r = item['r_to_r']
            train_g_list[i].r_to_e = item['r_to_e']
    with open(json_path.format(dataset, 'valid.json'), 'r') as f:
        info_dict = json.load(f)
        for i in range(len(valid_g_list)):
            item = info_dict['{}'.format(i)]
            # valid_g_list[i].uniq_e = item['uniq_e']
            valid_g_list[i].uniq_r = item['uniq_r']
            # valid_g_list[i].e_to_r = item['e_to_r']
            valid_g_list[i].r_to_r = item['r_to_r']
            valid_g_list[i].r_to_e = item['r_to_e']

    with open(json_path.format(dataset, 'test.json'), 'r') as f:
        info_dict = json.load(f)
        for i in range(len(test_g_list)):
            item = info_dict['{}'.format(i)]
            # test_g_list[i].uniq_e = item['uniq_e']
            test_g_list[i].uniq_r = item['uniq_r']
            # test_g_list[i].e_to_r = item['e_to_r']
            test_g_list[i].r_to_r = item['r_to_r']
            test_g_list[i].r_to_e = item['r_to_e']

    return train_g_list, valid_g_list, test_g_list
    # return train_list, valid_list, test_list, train_times, valid_times, test_times, num_ent, num_rel, num_time, train_g_list, valid_g_list, test_g_list
