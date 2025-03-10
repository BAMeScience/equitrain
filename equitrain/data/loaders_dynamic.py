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
                # Drop item which has more nodes than allowed
                continue

            if (
                self.max_nodes is not None
                and self.drop
                and item.num_nodes > self.max_nodes
            ):
                # Drop item which has more edges than allowed
                continue

            # Further checks only if there is at least one graph in the current batch
            if current_batch:
                if (
                    self.max_nodes is not None
                    and current_node_sum + item.num_nodes > self.max_nodes
                ):
                    # Maximum number of edges reached
                    dynamic_batches.append(self.collate_fn(current_batch))
                    current_batch = []
                    current_node_sum = 0
                    current_edge_sum = 0

                if (
                    self.max_edges is not None
                    and current_edge_sum + item.num_edges > self.max_edges
                ):
                    # Maximum number of edges reached
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
                errors[errors >= args.weighted_sampler_threshold] = 0.0
            # Errors might be zero, in case a sample was not used during training
            errors[errors == 0.0] = errors.mean()
            # Add sampler to the keyword arguments, use errors as weights
            kwargs['sampler'] = WeightedRandomSampler(
                errors, num_samples=len(errors), replacement=True, generator=generator
            )

        super().__init__(*args, **kwargs)

        self.collate_fn = DynamicGraphCollater(
            self.collate_fn, max_nodes=max_nodes, max_edges=max_edges, drop=drop
        )
