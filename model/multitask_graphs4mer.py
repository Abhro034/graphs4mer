"""
Multi-task GraphS4mer model for Sleep Stage Classification and ADHD Detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.graphs4mer import GraphS4mer


class MultiTaskGraphS4mer(nn.Module):
    """
    Multi-task model with separate heads for:
    1. Sleep stage classification (epoch-level, 5 classes: Wake, N1, N2, N3, REM)
    2. ADHD detection (patient-level, binary: 0 or 1)

    The model shares a common GraphS4mer backbone and has two separate classification heads.
    """
    def __init__(
        self,
        input_dim,
        num_nodes,
        dropout,
        num_temporal_layers,
        g_conv,
        num_gnn_layers,
        hidden_dim,
        max_seq_len,
        resolution,
        num_sleep_classes=5,  # Wake, N1, N2, N3, REM
        num_adhd_classes=2,   # 0 (non-ADHD), 1 (ADHD)
        state_dim=64,
        channels=1,
        temporal_model="s4",
        bidirectional=False,
        prenorm=False,
        postact=None,
        metric="self_attention",
        adj_embed_dim=10,
        gin_mlp=False,
        train_eps=False,
        prune_method="thresh",
        edge_top_perc=0.5,
        thresh=None,
        temporal_pool="mean",
        graph_pool="sum",
        activation_fn="relu",
        undirected_graph=True,
        use_prior=False,
        K=3,
        regularizations=["feature_smoothing", "degree", "sparse"],
        residual_weight=0.0,
        decay_residual_weight=False,
        **kwargs
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_sleep_classes = num_sleep_classes
        self.num_adhd_classes = num_adhd_classes
        self.graph_pool = graph_pool
        self.temporal_pool = temporal_pool

        # Shared backbone - GraphS4mer without classifier
        self.backbone = GraphS4mer(
            input_dim=input_dim,
            num_nodes=num_nodes,
            dropout=dropout,
            num_temporal_layers=num_temporal_layers,
            g_conv=g_conv,
            num_gnn_layers=num_gnn_layers,
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
            resolution=resolution,
            state_dim=state_dim,
            channels=channels,
            temporal_model=temporal_model,
            bidirectional=bidirectional,
            prenorm=prenorm,
            postact=postact,
            metric=metric,
            adj_embed_dim=adj_embed_dim,
            gin_mlp=gin_mlp,
            train_eps=train_eps,
            prune_method=prune_method,
            edge_top_perc=edge_top_perc,
            thresh=thresh,
            temporal_pool=temporal_pool,
            graph_pool=graph_pool,
            activation_fn=activation_fn,
            num_classes=1,  # dummy, we'll replace the classifier
            undirected_graph=undirected_graph,
            use_prior=use_prior,
            K=K,
            regularizations=regularizations,
            residual_weight=residual_weight,
            decay_residual_weight=decay_residual_weight,
            **kwargs
        )

        # Remove the default classifier from backbone
        del self.backbone.classifier

        # Task-specific heads
        self.sleep_stage_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_sleep_classes)
        )

        self.adhd_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, num_adhd_classes)
        )

    def get_shared_features(self, data, return_attention=False, lengths=None, epoch=None, epoch_total=None):
        """
        Extract shared features from the backbone network.

        Args:
            data: torch geometric data object
            return_attention: whether to return attention weights
            lengths: sequence lengths for variable-length inputs
            epoch: current epoch (for residual weight decay)
            epoch_total: total epochs (for residual weight decay)

        Returns:
            features: (batch, hidden_dim) shared representation
            reg_losses: regularization losses dictionary
            attention_outputs: (optional) attention weights
        """
        x = data.x  # (batch * num_nodes, seq_len, input_dim)
        batch = x.shape[0] // self.backbone.num_nodes
        num_nodes = self.backbone.num_nodes
        _, seq_len, _ = x.shape
        batch_idx = data.batch

        if lengths is not None:
            lengths = torch.repeat_interleave(lengths, num_nodes, dim=0)

        # Temporal layer
        if self.backbone.temporal_model == "s4":
            x = self.backbone.t_model(x, lengths)  # (batch * num_nodes, seq_len, hidden_dim)
        else:
            if lengths is not None:
                from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
                x = pack_padded_sequence(x, lengths, batch_first=True)
            x, _ = self.backbone.t_model(x)
            if lengths is not None:
                x, lengths = pad_packed_sequence(x, batch_first=True)

        # Get output with <resolution> as interval
        if lengths is None:
            x = x.view(batch, num_nodes, seq_len, -1)  # (batch, num_nodes, seq_len, hidden_dim)
            x_tmp = []
            num_dynamic_graphs = self.backbone.max_seq_len // self.backbone.resolution
            for t in range(num_dynamic_graphs):
                start = t * self.backbone.resolution
                stop = start + self.backbone.resolution
                curr_x = torch.mean(x[:, :, start:stop, :], dim=2)
                x_tmp.append(curr_x)
            x_tmp = torch.stack(x_tmp, dim=1)  # (batch, num_dynamic_graphs, num_nodes, hidden_dim)
            x = x_tmp.reshape(-1, num_nodes, self.hidden_dim)  # (batch * num_dynamic_graphs, num_nodes, hidden_dim)
            del x_tmp
        else:  # for variable lengths, mean pool over actual lengths
            x = torch.stack(
                [
                    torch.mean(out[:length, :], dim=0)
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
            x = x.reshape(batch, num_nodes, -1)  # (batch, num_nodes, hidden_dim)
            num_dynamic_graphs = 1

        # Get initial adj
        if self.backbone.use_prior:
            import torch_geometric
            adj_mat = torch_geometric.utils.to_dense_adj(
                edge_index=data.edge_index, batch=data.batch, edge_attr=data.edge_attr
            )
        else:
            # knn cosine graph
            from model.graphs4mer import get_knn_graph
            import torch_geometric
            edge_index, edge_weight, adj_mat = get_knn_graph(
                x,
                self.backbone.K,
                dist_measure="cosine",
                undirected=self.backbone.undirected_graph,
            )
            edge_index = edge_index.to(x.device)
            edge_weight = edge_weight.to(x.device)
            adj_mat = adj_mat.to(x.device)

        # Learn adj mat
        attn_weight = self.backbone.attn_layers(x)  # (batch*num_dynamic_graphs, num_nodes, num_nodes)

        # To undirected
        if self.backbone.undirected_graph:
            attn_weight = (attn_weight + attn_weight.transpose(1, 2)) / 2
        raw_attn_weight = attn_weight.clone()

        # Add residual
        if len(adj_mat.shape) == 2:
            adj_mat = torch.cat([adj_mat] * num_dynamic_graphs * batch, dim=0)
        elif len(adj_mat.shape) == 3 and (adj_mat.shape != attn_weight.shape):
            adj_mat = torch.cat([adj_mat] * num_dynamic_graphs, dim=0)

        # KNN graph weight (aka residual weight) decay
        if self.backbone.decay_residual_weight:
            from model.graphs4mer import calculate_cosine_decay_weight
            assert (epoch is not None) and (epoch_total is not None)
            residual_weight = calculate_cosine_decay_weight(
                max_weight=self.backbone.residual_weight, epoch=epoch, epoch_total=epoch_total, min_weight=0
            )
        else:
            residual_weight = self.backbone.residual_weight
        # Add knn graph
        adj_mat = residual_weight * adj_mat + (1 - residual_weight) * attn_weight

        # Prune graph
        from model.graphs4mer import prune_adj_mat
        adj_mat = prune_adj_mat(
            adj_mat,
            num_nodes,
            method=self.backbone.prune_method,
            edge_top_perc=self.backbone.edge_top_perc,
            knn=self.backbone.K,
            thresh=self.backbone.thresh,
        )

        # Regularization loss
        reg_losses = self.backbone.regularization_loss(x, adj=adj_mat)

        # Back to sparse graph
        import torch_geometric
        edge_index, edge_weight = torch_geometric.utils.dense_to_sparse(adj_mat)

        # Add self-loop
        edge_index, edge_weight = torch_geometric.utils.remove_self_loops(
            edge_index=edge_index, edge_attr=edge_weight
        )
        edge_index, edge_weight = torch_geometric.utils.add_self_loops(
            edge_index=edge_index,
            edge_attr=edge_weight,
            fill_value=1,
        )

        x = x.view(batch * num_dynamic_graphs * num_nodes, -1)  # (batch * num_dynamic_graphs * num_nodes, hidden_dim)
        for i in range(len(self.backbone.gnn_layers)):
            # GNN layer
            x = self.backbone.gnn_layers[i](
                x, edge_index=edge_index, edge_attr=edge_weight.reshape(-1, 1)
            )
            x = self.backbone.dropout(self.backbone.activation(x))
        x = x.view(batch * num_dynamic_graphs, num_nodes, -1).view(
            batch, num_dynamic_graphs, num_nodes, -1
        )  # (batch, num_dynamic_graphs, num_nodes, hidden_dim)

        # Temporal pool
        if self.temporal_pool == "last":
            x = x[:, -1, :, :]  # (batch, num_nodes, hidden_dim)
        elif self.temporal_pool == "mean":
            x = torch.mean(x, dim=1)
        else:
            raise NotImplementedError

        # Graph pool
        if self.graph_pool == "sum":
            x = torch.sum(x, dim=1)  # (batch, hidden_dim)
        elif self.graph_pool == "mean":
            x = torch.mean(x, dim=1)
        elif self.graph_pool == "max":
            x, _ = torch.max(x, dim=1)
        else:
            raise NotImplementedError

        feat = x.clone()

        if return_attention:
            return (
                feat,
                reg_losses,
                raw_attn_weight.reshape(batch, num_dynamic_graphs, num_nodes, num_nodes),
                adj_mat.reshape(batch, num_dynamic_graphs, num_nodes, num_nodes),
            )
        else:
            return feat, reg_losses

    def forward(self, data, task="both", return_attention=False, lengths=None, epoch=None, epoch_total=None):
        """
        Forward pass for multi-task learning.

        Args:
            data: torch geometric data object
            task: which task to perform - "sleep", "adhd", or "both"
            return_attention: whether to return attention weights
            lengths: sequence lengths for variable-length inputs
            epoch: current epoch
            epoch_total: total epochs

        Returns:
            If task == "both":
                (sleep_logits, adhd_logits, reg_losses)
            If task == "sleep":
                (sleep_logits, reg_losses)
            If task == "adhd":
                (adhd_logits, reg_losses)
        """
        # Get shared features
        if return_attention:
            features, reg_losses, raw_attn, adj_learned = self.get_shared_features(
                data, return_attention=True, lengths=lengths, epoch=epoch, epoch_total=epoch_total
            )
        else:
            features, reg_losses = self.get_shared_features(
                data, return_attention=False, lengths=lengths, epoch=epoch, epoch_total=epoch_total
            )

        # Task-specific predictions
        if task == "both":
            sleep_logits = self.sleep_stage_head(features)  # (batch, num_sleep_classes)
            adhd_logits = self.adhd_head(features)  # (batch, num_adhd_classes)

            if return_attention:
                return sleep_logits, adhd_logits, reg_losses, raw_attn, adj_learned, features
            else:
                return sleep_logits, adhd_logits, reg_losses

        elif task == "sleep":
            sleep_logits = self.sleep_stage_head(features)  # (batch, num_sleep_classes)

            if return_attention:
                return sleep_logits, reg_losses, raw_attn, adj_learned, features
            else:
                return sleep_logits, reg_losses

        elif task == "adhd":
            adhd_logits = self.adhd_head(features)  # (batch, num_adhd_classes)

            if return_attention:
                return adhd_logits, reg_losses, raw_attn, adj_learned, features
            else:
                return adhd_logits, reg_losses
        else:
            raise ValueError(f"Invalid task: {task}. Must be 'sleep', 'adhd', or 'both'")
