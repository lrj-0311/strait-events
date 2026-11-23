"""Data-free checks for the graph and scoring primitives."""

import importlib.util
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decoder import ConvTrans_decoder
from layers import R_GCN
from utils import build_sub_graph


class GraphCoreTests(unittest.TestCase):
    def test_inverse_edges_and_degree_weights(self):
        facts = np.array([[0, 0, 1, 0], [0, 1, 2, 0]], dtype=np.int64)
        graph = build_sub_graph(4, 2, None, facts, torch.device("cpu"))
        source, target = graph.edges(order="eid")
        self.assertEqual(source.tolist(), [0, 0, 1, 2])
        self.assertEqual(target.tolist(), [1, 2, 0, 0])
        self.assertEqual(graph.edata["type"].tolist(), [0, 1, 2, 3])
        self.assertEqual(graph.ndata["in_norm"].view(-1).tolist(), [0.5, 1.0, 1.0, 1.0])
        self.assertEqual(graph.edata["up_norm"].view(-1).tolist(), [0.5, 0.5, 1.0, 1.0])

    def test_message_aggregation_and_gradients(self):
        facts = np.array([[0, 0, 1, 0], [0, 1, 2, 0]], dtype=np.int64)
        graph = build_sub_graph(4, 2, None, facts, torch.device("cpu"))
        entities = torch.tensor([[1.0, 2.0], [2.0, 1.0], [1.0, 1.0], [3.0, 2.0]], requires_grad=True)
        relations = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]], requires_grad=True)
        layer = R_GCN(2, 2, dropout=0.0, activation=torch.relu)
        with torch.no_grad():
            layer.weight_neighbor.copy_(torch.eye(2))
            layer.loop_weight.copy_(2 * torch.eye(2))
            layer.evolve_loop_weight.copy_(3 * torch.eye(2))

        output = layer(graph, entities, relations)
        expected = torch.tensor([[4.5, 6.0], [6.0, 4.0], [3.0, 5.0], [9.0, 6.0]])
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        self.assertTrue(torch.isfinite(entities.grad).all())
        self.assertTrue(torch.isfinite(relations.grad).all())
        self.assertTrue(torch.isfinite(layer.weight_neighbor.grad).all())

    def test_edgeless_graph_uses_evolution_state(self):
        graph = build_sub_graph(1, 1, None, np.empty((0, 4), dtype=np.int64), torch.device("cpu"))
        layer = R_GCN(2, 2, dropout=0.0)
        with torch.no_grad():
            layer.evolve_loop_weight.copy_(torch.eye(2))
        output = layer(graph, torch.tensor([[1.0, 2.0]]), torch.zeros(2, 2))
        torch.testing.assert_close(output, torch.tensor([[1.0, 2.0]]))

    def test_legacy_graph_outputs_match_when_reference_available(self):
        reference_dir = os.environ.get("MUSEED_REFERENCE_SRC")
        if not reference_dir:
            self.skipTest("Set MUSEED_REFERENCE_SRC to compare with the earlier implementation")

        def load_reference(name):
            spec = importlib.util.spec_from_file_location(f"reference_{name}", Path(reference_dir) / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

        reference_utils = load_reference("utils")
        reference_layers = load_reference("layers")
        facts = np.array([[0, 0, 1, 0], [0, 1, 2, 0]], dtype=np.int64)
        old_graph = reference_utils.build_sub_graph(4, 2, None, facts, torch.device("cpu"))
        new_graph = build_sub_graph(4, 2, None, facts, torch.device("cpu"))
        for field in ("id", "in_norm", "out_norm"):
            torch.testing.assert_close(new_graph.ndata[field], old_graph.ndata[field])
        for field in ("type", "up_norm"):
            torch.testing.assert_close(new_graph.edata[field], old_graph.edata[field])

        old_layer = reference_layers.R_GCN(3, 3, dropout=0.0, activation=torch.relu).eval()
        new_layer = R_GCN(3, 3, dropout=0.0, activation=torch.relu).eval()
        new_layer.load_state_dict(old_layer.state_dict(), strict=True)
        entities = torch.randn(4, 3)
        relations = torch.randn(4, 3)
        torch.testing.assert_close(new_layer(new_graph, entities, relations),
                                   old_layer(old_graph, entities, relations))


class DecoderTests(unittest.TestCase):
    def test_branch_scores_form_probability_distribution(self):
        decoder = ConvTrans_decoder(4, 4, torch.device("cpu"), channels=2, d_k=2, d_v=2, n_heads=1, d_ff=8)
        decoder.eval()
        facts = torch.tensor([[0, 0, 1], [1, 1, 2]])
        entities = [torch.randn(4, 4)]
        relations = [torch.randn(2, 4)]
        history = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])
        scores, parts = decoder(facts, entities, relations, [None], one_hot_tail_seq=history,
                                freq_info=1, return_components=True)
        combined = sum(parts.values())
        torch.testing.assert_close(scores.exp(), combined)
        torch.testing.assert_close(combined.sum(dim=1), torch.ones(2, dtype=combined.dtype))

    def test_legacy_decoder_outputs_match_when_reference_available(self):
        reference_dir = os.environ.get("MUSEED_REFERENCE_SRC")
        if not reference_dir:
            self.skipTest("Set MUSEED_REFERENCE_SRC to compare with the earlier implementation")
        spec = importlib.util.spec_from_file_location("reference_decoder", Path(reference_dir) / "decoder.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        kwargs = dict(channels=2, d_k=2, d_v=2, n_heads=1, d_ff=8, alpha=0.65)
        reference = module.ConvTrans_decoder(4, 4, torch.device("cpu"), **kwargs).eval()
        current = ConvTrans_decoder(4, 4, torch.device("cpu"), **kwargs).eval()
        current.load_state_dict(reference.state_dict(), strict=True)
        facts = torch.tensor([[0, 0, 1], [1, 1, 2]])
        entities = [torch.randn(4, 4)]
        relations = [torch.randn(2, 4)]
        local_bias = torch.randn(4, 4)
        context = torch.randn(2, 4)
        history = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])
        cases = [
            (None, 0, None, 0.0),
            (None, 1, None, 0.0),
            (local_bias, 0, context, 0.0),
            (local_bias, 1, context, 0.03),
        ]
        for bias, frequency, context_state, rule_weight in cases:
            options = dict(one_hot_tail_seq=history, freq_info=frequency,
                           query_context_state=context_state, rule_branch_weight=rule_weight,
                           return_components=True, return_auxiliary=True)
            old_score, old_parts, old_aux = reference(facts, entities, relations, [bias], **options)
            new_score, new_parts, new_aux = current(facts, entities, relations, [bias], **options)
            torch.testing.assert_close(new_score, old_score)
            for name in old_parts:
                torch.testing.assert_close(new_parts[name], old_parts[name])
            torch.testing.assert_close(new_aux["historical_logits"], old_aux["historical_logits"])
        torch.testing.assert_close(current.pretrain_predict(facts, entities[0], relations[0]),
                                   reference.pretrain_predict(facts, entities[0], relations[0]))
        negative_facts = torch.tensor([[0, 0, 2], [1, 1, 3]])
        torch.testing.assert_close(current.pretrain_loss(facts, negative_facts, entities[0], relations[0]),
                                   reference.pretrain_loss(facts, negative_facts, entities[0], relations[0]))


if __name__ == "__main__":
    unittest.main()
