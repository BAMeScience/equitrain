from __future__ import annotations

import torch_geometric
from torch.utils.data import WeightedRandomSampler


class DynamicGraphCollater:
    def __init__(self, collate_fn, max_nodes=None, max_edges=None, drop=False):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.drop = drop
        self.collate_fn = collate_fn

    def __call__(self, batch):
        dynamic_batches = []
        current_batch = []
        current_node_sum = 0
        current_edge_sum = 0

        for item in batch:
            if (
                self.max_edges is not None
                and self.drop
                and item.num_edges > self.max_edges
            ):
                continue

            if (
                self.max_nodes is not None
                and self.drop
                and item.num_nodes > self.max_nodes
            ):
                continue

            if current_batch:
                if (
                    self.max_nodes is not None
                    and current_node_sum + item.num_nodes > self.max_nodes
                ):
                    dynamic_batches.append(self.collate_fn(current_batch))
                    current_batch = []
                    current_node_sum = 0
                    current_edge_sum = 0

                if (
                    self.max_edges is not None
                    and current_edge_sum + item.num_edges > self.max_edges
                ):
                    dynamic_batches.append(self.collate_fn(current_batch))
                    current_batch = []
                    current_node_sum = 0
                    current_edge_sum = 0

            current_batch.append(item)
            current_node_sum += item.num_nodes
            current_edge_sum += item.num_edges

        if current_batch:
            dynamic_batches.append(self.collate_fn(current_batch))

        return dynamic_batches


class DynamicGraphLoader(torch_geometric.loader.DataLoader):
    def __init__(
        self,
        *args,
        errors=None,
        errors_threshold=None,
        max_nodes=None,
        max_edges=None,
        drop=False,
        generator=None,
        **kwargs,
    ):
        if errors is not None:
            if errors_threshold is not None:
                errors[errors >= errors_threshold] = 0.0
            errors[errors == 0.0] = errors.mean()
            kwargs['sampler'] = WeightedRandomSampler(
                errors, num_samples=len(errors), replacement=True, generator=generator
            )

        super().__init__(*args, **kwargs)

        self.collate_fn = DynamicGraphCollater(
            self.collate_fn, max_nodes=max_nodes, max_edges=max_edges, drop=drop
        )
