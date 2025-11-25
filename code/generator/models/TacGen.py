from typing import Union, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Module):
    """
    Args:
        input_dims (int): Dimension of input.
        output_dims (int): Dimension of output.
        kernel_size (tuple or list): Size of the convolution kernel.
        stride (tuple or list, optional): Convolution strides, default (1,1).
        use_bias (bool, optional): Whether to use bias, default is True.
        activation (Callable, optional): Activation function, default is torch.nn.functional.relu.
    """

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        kernel_size: Union[tuple, list],
        stride: Union[tuple, list] = (1, 1),
        use_bias: bool = True,
        activation: Optional[Callable[[torch.FloatTensor], torch.FloatTensor]] = F.relu,
    ):
        super(Conv2D, self).__init__()
        self._activation = activation
        self._conv2d = nn.Conv2d(
            input_dims,
            output_dims,
            kernel_size,
            stride=stride,
            padding=0,
            bias=use_bias,
        )
        # Initialization
        torch.nn.init.xavier_uniform_(self._conv2d.weight)
        if use_bias:
            torch.nn.init.zeros_(self._conv2d.bias)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the 2D-convolution block.
        Arg types:
            * **X** (PyTorch Float Tensor) - Input tensor, with shape (batch_size, num_his, num_nodes, input_dims).
        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, num_his, num_nodes, output_dims).
        """
        X = X.permute(0, 3, 2, 1)
        X = self._conv2d(X)
        if self._activation is not None:
            X = self._activation(X)
        return X.permute(0, 3, 2, 1)


class FullyConnected(nn.Module):
    """
    Args:
        input_dims (int or list): Dimension(s) of input.
        units (int or list): Dimension(s) of outputs in each 2D convolution block.
        activations (Callable or list): Activation function(s).
        use_bias (bool, optional): Whether to use bias, default is True.
    """

    def __init__(
        self,
        input_dims: Union[int, list],
        units: Union[int, list],
        activations: Union[Callable[[torch.FloatTensor], torch.FloatTensor], list],
        use_bias: bool = True,
    ):
        super(FullyConnected, self).__init__()
        if isinstance(units, int):
            units = [units]
            input_dims = [input_dims]
            activations = [activations]
        assert type(units) == list
        self._conv2ds = nn.ModuleList(
            [
                Conv2D(
                    input_dims=input_dim,
                    output_dims=num_unit,
                    kernel_size=[1, 1],
                    stride=[1, 1],
                    use_bias=use_bias,
                    activation=activation,
                )
                for input_dim, num_unit, activation in zip(
                    input_dims, units, activations
                )
            ]
        )

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        """
        Making a forward pass of the fully-connected layer.
        Arg types:
            * **X** (PyTorch Float Tensor) - Input tensor, with shape (batch_size, num_his, num_nodes, 1).
        Return types:
            * **X** (PyTorch Float Tensor) - Output tensor, with shape (batch_size, num_his, num_nodes, units[-1]).
        """
        for conv in self._conv2ds:
            X = conv(X)
        return X


class DynamicAttentionPooling(nn.Module):
    def __init__(self, embedding_dim=768):
        super().__init__()
        self._weights = nn.Linear(embedding_dim, 1)
        self.attn = nn.MultiheadAttention(embedding_dim, num_heads=4, batch_first=True)

    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        # return: [batch_size, embedding_dim]
        attn_output, _ = self.attn(x, x, x)
        weights = F.softmax(self._weights(attn_output), dim=1)  # [batch, seq_len, 1]
        pooled_output = torch.sum(attn_output * weights, dim=1)
        return pooled_output


class TimeLangEmbedding(nn.Module):
    r"""
    Args:
        D (int) : Dimension of output.
        use_bias (bool, optional): Whether to use bias in Fully Connected layers, default is True.
    """

    def __init__(
        self,
        D: int,
        D_time: int = 128,
        D_lang: int = 768,
        use_bias: bool = True
    ):
        super(TimeLangEmbedding, self).__init__()

        self._fully_connected_te = FullyConnected(
            input_dims=[D_time, D],
            units=[D, D],
            activations=[F.relu, None],
            use_bias=use_bias,
        )

        self._fully_connected_le = FullyConnected(
            input_dims=[D_lang, D],
            units=[D, D],
            activations=[F.relu, None],
            use_bias=use_bias,
        )
        self._dynamic_att_pool = DynamicAttentionPooling()

    def forward(
        self, TE: torch.FloatTensor, LE: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the spatial-temporal embedding.
        Arg types:
            * **TE** (Pytorch Float Tensor) - Temporal embedding, with shape (batch_size, T, D_time).
            * **LE** (Pytorch Float Tensor) - Language embedding, with shape (batch_size, word_len, D_lang).
        Return types:
            * **output** (PyTorch Float Tensor) - Spatial-temporal embedding, with shape (batch_size, T, num_nodes, D).
        """
        batch_size, T = TE.shape[0], TE.shape[1]
        TE = TE.to(LE.device)
        TE = TE.unsqueeze(2)  # (batch_size, T, 1, D_time)
        TE = self._fully_connected_te(TE)

        LE = self._dynamic_att_pool(LE)
        LE = LE.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, D_lang)
        LE = self._fully_connected_le(LE)

        TE = TE.repeat(1, 1, 23, 1)
        LE = LE.repeat(1, T, 23, 1)
        return torch.concat((TE, LE), dim=-1)


class SpatialAttention(nn.Module):
    """
    Args:
        K (int) : Number of attention heads.
        d (int) : Dimension of each attention head outputs.
    """

    def __init__(self, K: int, d: int):
        super(SpatialAttention, self).__init__()
        D = K * d
        self._d = d
        self._K = K
        self._fully_connected_q = FullyConnected(input_dims=3 * D, units=D, activations=F.relu)
        self._fully_connected_k = FullyConnected(input_dims=3 * D, units=D, activations=F.relu)
        self._fully_connected_v = FullyConnected(input_dims=3 * D, units=D, activations=F.relu)
        self._fully_connected = FullyConnected(input_dims=[D, D], units=[D, D], activations=[F.relu, None])

    def forward(
        self, X: torch.FloatTensor, TLE: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the spatial attention mechanism.
        Arg types:
            * **X** (PyTorch Float Tensor) - Input sequence, with shape (batch_size, num_step, num_nodes, K*d).
            * **TLE** (Pytorch Float Tensor) - Spatial-temporal embedding, with shape (batch_size, num_step, num_nodes, K*d).
        Return types:
            * **X** (PyTorch Float Tensor) - Spatial attention scores, with shape (batch_size, num_step, num_nodes, K*d).
        """
        batch_size = X.shape[0]
        X = torch.cat((X, TLE), dim=-1)
        query = self._fully_connected_q(X)  # (B, T, N, D)
        key = self._fully_connected_k(X)
        value = self._fully_connected_v(X)
        query = torch.cat(torch.split(query, self._d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self._d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self._d, dim=-1), dim=0)
        attention = torch.matmul(query, key.transpose(2, 3))
        attention /= self._d ** 0.5
        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        del query, key, value, attention
        X = self._fully_connected(X)  # (B, T, N, D)
        return X


class TemporalAttention(nn.Module):
    """
    Args:
        K (int) : Number of attention heads.
        d (int) : Dimension of each attention head outputs.
        mask (bool): Whether to mask attention score.
    """

    def __init__(self, K: int, d: int, mask: bool):
        super(TemporalAttention, self).__init__()
        D = K * d
        self._d = d
        self._K = K
        self._mask = mask
        self._fully_connected_q = FullyConnected(input_dims=3 * D, units=D, activations=F.relu)
        self._fully_connected_k = FullyConnected(input_dims=3 * D, units=D, activations=F.relu)
        self._fully_connected_v = FullyConnected(input_dims=3 * D, units=D, activations=F.relu)
        self._fully_connected = FullyConnected(input_dims=[D, D], units=[D, D], activations=[F.relu, None])

    def forward(
        self, X: torch.FloatTensor, TLE: torch.FloatTensor, key_padding_mask
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the temporal attention mechanism.

        Arg types:
            * **X** (PyTorch Float Tensor) - Input sequence, with shape (batch_size, num_step, num_nodes, K*d).
            * **TLE** (Pytorch Float Tensor) - Spatial-temporal embedding, with shape (batch_size, num_step, num_nodes, K*d).
            * **key_padding_mask** (Pytorch Int Tensor) - padding length with shape (batch_size, 1).

        Return types:
            * **X** (PyTorch Float Tensor) - Temporal attention scores, with shape (batch_size, num_step, num_nodes, K*d).
        """
        batch_size, num_step, num_nodes = X.shape[0], X.shape[1], X.shape[2]
        X = torch.cat((X, TLE), dim=-1)
        query = self._fully_connected_q(X)
        key = self._fully_connected_k(X)
        value = self._fully_connected_v(X)
        query = torch.cat(torch.split(query, self._d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self._d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self._d, dim=-1), dim=0)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        score = torch.matmul(query, key)
        score /= self._d ** 0.5
        if self._mask:
            batch_size, num_step, num_nodes = X.shape[0], X.shape[1], X.shape[2]
            mask = torch.tril(torch.ones(num_step, num_step, device=score.device)).bool()
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.repeat(self._K * batch_size, num_nodes, 1, 1)  # (batch_size * K, num_nodes, num_step, num_step)
            score = score.masked_fill(~mask, float('-inf'))
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat(self._K, 1)  # (batch_size * K, 1)
            arange = torch.arange(num_step, device=score.device).unsqueeze(0)  # (1, num_step)
            padding_mask = (arange >= key_padding_mask)  # (batch_size * K, num_step)
            score = score.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        score = F.softmax(score, dim=-1)
        attention = torch.matmul(score, value)
        attention = attention.permute(0, 2, 1, 3)
        attention = torch.cat(torch.split(attention, batch_size, dim=0), dim=-1)
        del query, key, value
        attention = self._fully_connected(attention)  # (B, T, N, D)
        return attention

class GatedFusion(nn.Module):
    """
    Args:
        D (int) : dimension of output.
    """

    def __init__(self, D: int):
        super(GatedFusion, self).__init__()
        self._fully_connected_xs = FullyConnected(input_dims=D, units=D, activations=None, use_bias=False)
        self._fully_connected_xt = FullyConnected(input_dims=D, units=D, activations=None, use_bias=True)
        self._fully_connected_h = FullyConnected(input_dims=[D, D], units=[D, D], activations=[F.relu, None])

    def forward(
        self, HS: torch.FloatTensor, HT: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the gated fusion mechanism.
        Arg types:
            * **HS** (PyTorch Float Tensor) - Spatial attention scores, with shape (batch_size, num_step, num_nodes, D).
            * **HT** (Pytorch Float Tensor) - Temporal attention scores, with shape (batch_size, num_step, num_nodes, D).
        Return types:
            * **H** (PyTorch Float Tensor) - Spatial-temporal attention scores, with shape (batch_size, num_step, num_nodes, D).
        """
        XS = self._fully_connected_xs(HS)
        XT = self._fully_connected_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self._fully_connected_h(H)
        del XS, XT, z
        return H


class SpatioTemporalAttention(nn.Module):
    """Args:
        K (int) : Number of attention heads.
        d (int) : Dimension of each attention head outputs.
        mask (bool): Whether to mask attention score in temporal attention.
    """

    def __init__(self, K: int, d: int, mask: bool):
        super(SpatioTemporalAttention, self).__init__()
        self._spatial_attention = SpatialAttention(K, d)
        self._temporal_attention = TemporalAttention(K, d, mask=mask)
        self._gated_fusion = GatedFusion(K * d)

    def forward(
        self, X: torch.FloatTensor, TLE: torch.FloatTensor, key_padding_mask
    ) -> torch.FloatTensor:
        """
        Making a forward pass of the spatial-temporal attention block.
        Arg types:
            * **X** (PyTorch Float Tensor) - Input sequence, with shape (batch_size, num_step, num_nodes, K*d).
            * **TLE** (Pytorch Float Tensor) - Spatial-temporal embedding, with shape (batch_size, num_step, num_nodes, K*d).
        Return types:
            * **X** (PyTorch Float Tensor) - Attention scores, with shape (batch_size, num_step, num_nodes, K*d).
        """
        HS = self._spatial_attention(X, TLE)
        HT = self._temporal_attention(X, TLE, key_padding_mask)
        H = self._gated_fusion(HS, HT)
        del HS, HT
        X = torch.add(X, H)
        return X


class CrossAttention(nn.Module):
    def __init__(self, K: int, d: int):
        super().__init__()
        D = K * d
        self._K = K
        self._d = d
        self._fully_connected_q = FullyConnected(input_dims=D, units=D, activations=F.relu)
        self._fully_connected_k = FullyConnected(input_dims=D, units=D, activations=F.relu)
        self._fully_connected_v = FullyConnected(input_dims=D, units=D, activations=F.relu)

    def forward(self, hidden_state, decoder_embedding):
        """
        Making a forward pass of the cross attention mechanism.
        Arg types:
            * **encoder_embedding** (PyTorch Float Tensor) - Encoder output, with shape (batch_size, num_nodes, K*d).
            * **decoder_embedding** (PyTorch Float Tensor) - Decoder input, with shape (batch_size, num_nodes, K*d).
        Return types:
            * **X** (PyTorch Float Tensor) - Cross attention scores, with shape (batch_size, num_nodes, K*d).
        """
        batch_size = hidden_state.shape[0]
        query = self._fully_connected_q(decoder_embedding)
        key = self._fully_connected_k(hidden_state)
        value = self._fully_connected_v(hidden_state)

        query = torch.cat(torch.split(query, self._d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self._d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self._d, dim=-1), dim=0)

        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)

        attention = torch.matmul(query, key)
        attention /= self._d ** 0.5
        attention = F.softmax(attention, dim=-1)

        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        del query, key, value, attention

        return X


class EncoderLayer(nn.Module):
    def __init__(self, K, d):
        super().__init__()
        self._sta = SpatioTemporalAttention(K, d, mask=False)
        self._fully_connected = FullyConnected(input_dims=[K * d, K * d], units=[K * d, K * d], activations=[F.relu, None])
        self._layer_norm_1 = nn.LayerNorm(K * d)
        self._layer_norm_2 = nn.LayerNorm(K * d)

    def forward(self, src, TLE_src, src_key_padding_mask):
        src = self._layer_norm_1(src + self._sta(src, TLE_src, src_key_padding_mask))
        src = self._layer_norm_2(src + self._fully_connected(src))
        return src

class DecoderLayer(nn.Module):
    def __init__(self, K, d):
        super().__init__()
        self._sta = SpatioTemporalAttention(K, d, mask=True)
        self._cross_attention = CrossAttention(K, d)
        self._fully_connected = FullyConnected(input_dims=[K * d, K * d], units=[K * d, K * d], activations=[F.relu, None])
        self._layer_norm_1 = nn.LayerNorm(K * d)
        self._layer_norm_2 = nn.LayerNorm(K * d)
        self._layer_norm_3 = nn.LayerNorm(K * d)

    def forward(self, hidden, tgt, TLE_tgt, tgt_key_padding_mask):
        tgt = self._layer_norm_1(tgt + self._sta(tgt, TLE_tgt, tgt_key_padding_mask))
        tgt = self._layer_norm_2(tgt + self._cross_attention(hidden, tgt))
        tgt = self._layer_norm_3(tgt + self._fully_connected(tgt))
        return tgt


class TacGen(nn.Module):

    def __init__(
        self,
        L: int,
        K: int,
        d: int,
        in_dim: int,
        num_his: int,
        num_pre: int,
        use_bias: bool,
    ):
        super().__init__()
        D = K * d
        self._num_his = num_his
        self._num_pre = num_pre
        self._position_encoding = PositionEncoding(d_model=D, max_len=num_pre)
        self._tl_embedding = TimeLangEmbedding(D, D_time=D, D_lang=768, use_bias=use_bias)
        self._encoder_layers = nn.ModuleList([EncoderLayer(K, d) for _ in range(L)])
        self._decoder_layers = nn.ModuleList([DecoderLayer(K, d) for _ in range(L)])
        # self._transform_attention = TransformAttention(K, d)
        self._cross_attention = CrossAttention(K, d)
        self._fully_connected_input = FullyConnected(input_dims=[in_dim, D], units=[D, D], activations=[F.relu, None])
        self._fully_connected_output_pos = FullyConnected(input_dims=[D, D], units=[D, 2], activations=[F.relu, None])
        self._fully_connected_output_stop = FullyConnected(input_dims=[D, D], units=[D, 1], activations=[F.relu, None])
        self.fc_mu = nn.Linear(D, D)
        self.fc_logvar = nn.Linear(D, D)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def use_cuda(self, device):
        self.to(device)  # 将整个模型移动到 GPU
        for name, module in self.named_children():
            module.to(device)  # 遍历所有子模块并移动到 GPU

    def forward(
        self,
        src: torch.Tensor,  # (batch_size, num_his, num_nodes, 2/待定)
        LE: torch.Tensor,  # (batch_size, word_len, 768)
        tgt: torch.Tensor,  # (batch_size, num_pre, num_nodes, 2/待定)
        TE_src=None,  # (batch_size, num_his, 3)
        TE_tgt=None,  # (batch_size, num_pre, 3)
        src_key_padding_mask=None,  # (batch_size, 1)
        tgt_key_padding_mask=None,  # (batch_size, 1)
    ):
        X_src = self._fully_connected_input(src)
        if TE_src is None:
            TE_src = self._position_encoding(src)
        if TE_tgt is None:
            TE_tgt = self._position_encoding(tgt)
        TLE_src = self._tl_embedding(TE_src, LE)
        TLE_tgt = self._tl_embedding(TE_tgt, LE)

        for net in self._encoder_layers:
            X_src = net(X_src, TLE_src, src_key_padding_mask)
        H = X_src

        X_tgt = self._fully_connected_input(tgt)
        for net in self._decoder_layers:
            X_tgt = net(H, X_tgt, TLE_tgt, tgt_key_padding_mask)
        pos = self._fully_connected_output_pos(X_tgt)
        stop = self._fully_connected_output_stop(X_tgt)
        stop = stop.squeeze(-1).mean(dim=-1)  # logits: (batch_size, T). NEED: please inference with Sigmoid
        return pos, stop

    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.
        Returns:
            int: Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def inference(
        self,
        src: torch.Tensor,  # (batch_size, num_his, num_nodes, input_dim)
        LE: torch.Tensor,   # (batch_size, word_len, 768)
        max_steps: int = None,
        stop_thresh: float = 0.1,
        device='cuda:0'
    ):
        """
        Autoregressive inference for trajectory generation.

        Args:
            src: Source sequence (batch_size, num_his, num_nodes, input_dim)
            LE: Language embedding (batch_size, word_len, 768)
            max_steps: Maximum prediction steps (default: self._num_pre)
            device: Device to run inference on

        Returns:
            predicted_pos: Generated positions (batch_size, pred_steps, num_nodes, 2)
            predicted_stop: Stop predictions (batch_size, pred_steps)
            actual_lengths: Actual sequence lengths for each batch (batch_size,)
        """
        self.eval()
        if max_steps is None:
            max_steps = self._num_pre

        batch_size, num_his, num_nodes, input_dim = src.shape

        # Initialize outputs
        predicted_pos = []
        predicted_stop = []
        actual_lengths = torch.full((batch_size,), max_steps, dtype=torch.long, device=device)
        already_stopped = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Initialize target sequence with the last step of source
        tgt = src[:, -1:, :, :]  # (batch_size, 1, num_nodes, input_dim)

        # 轨迹历史，用于平滑和约束
        trajectory_history = []

        with torch.no_grad():
            for step in range(max_steps):
                pos, stop = self.forward(
                    src=src,
                    LE=LE,
                    tgt=tgt,
                )

                # Get the latest prediction (last time step)
                next_pos = pos[:, -1:, :, :]  # (batch_size, 1, num_nodes, 2)
                next_stop = stop[:, -1:]      # (batch_size, 1)

                # Store predictions
                predicted_pos.append(next_pos)
                predicted_stop.append(next_stop)

                # Check stopping condition
                stop_probs = torch.sigmoid(next_stop)  # Convert logits to probabilities
                should_stop = stop_probs > stop_thresh  # (batch_size, 1)

                # Update actual lengths for sequences that should stop
                for b in range(batch_size):
                    if should_stop[b, 0] and actual_lengths[b] == max_steps:
                        actual_lengths[b] = step + 1
                        already_stopped[b] = True
                        print(f'Sequence {b} stopped at step {step + 1} with stop_prob {stop_probs[b, 0].item():.4f}')

                # Early stopping if all sequences should stop
                if already_stopped.all():
                    break

                # Prepare next input: use predicted position as next target input
                # Create next target input with same structure as original input
                next_input = torch.zeros_like(tgt[:, -1:, :, :])
                next_input[:, :, :, :2] = next_pos  # Use predicted position

                # Append to target sequence
                tgt = torch.cat([tgt, next_input], dim=1)
                # print(tgt[0, :, 4, :]*19.6+2.7)

        # Concatenate all predictions
        predicted_pos = torch.cat(predicted_pos, dim=1)  # (batch_size, pred_steps, num_nodes, 2)
        predicted_stop = torch.cat(predicted_stop, dim=1)  # (batch_size, pred_steps)
        return predicted_pos, predicted_stop, actual_lengths

    def inference_sample_point(
        self,
        src: torch.Tensor,  # (batch_size, num_his, num_nodes, input_dim)
        LE: torch.Tensor,   # (batch_size, word_len, 768)
        out_points: int = None,
        device='cuda:0'
    ):
        self.eval()
        predicted_pos = []
        tgt = src[:, -1:, :, :]  # (batch_size, 1, num_nodes, input_dim)
        with torch.no_grad():
            for step in range(out_points):
                pos, stop = self.forward(
                    src=src,
                    LE=LE,
                    tgt=tgt,
                )
                next_pos = pos[:, -1:, :, :]  # (batch_size, 1, num_nodes, 2)
                predicted_pos.append(next_pos)
                next_input = next_pos  # Use predicted position
                tgt = torch.cat([tgt, next_input], dim=1)

        # Concatenate all predictions
        predicted_pos = torch.cat(predicted_pos, dim=1)  # (batch_size, pred_steps, num_nodes, 2)
        return predicted_pos

class LossCriterion(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, out_pos, label_pos, out_stop, label_stop, mask_pos):
        """
        returns:
            * **loss** (float): Combined loss value.
            * **mse_loss** (float): Mean Squared Error loss for position prediction.
            * **bce_loss** (float): Binary Cross Entropy loss for stop prediction.
        """
        mse_loss = F.mse_loss(input=out_pos, target=label_pos, reduction='none')
        # bce_loss = F.binary_cross_entropy_with_logits(input=out_stop, target=label_stop, reduction='none', pos_weight=10*torch.ones_like(label_stop))
        bce_loss = F.binary_cross_entropy_with_logits(input=out_stop, target=label_stop, reduction='none')
        if mask_pos.sum()>0:
            mse_loss = mse_loss * mask_pos.float()
        mse_loss = mse_loss.mean()
        bce_loss = bce_loss.mean()
        return mse_loss + self.beta * bce_loss, mse_loss, bce_loss

class LossCriterion2(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, out_pos, label_pos):
        mse_loss = F.mse_loss(input=out_pos, target=label_pos, reduction='mean')
        return mse_loss

class PositionEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for sequence inputs.
    Usage:
        pe = PositionEncoding(d_model)
        pos_enc = pe(src)  # src: (batch_size, seq_len, ...)
    """
    def __init__(self, d_model, max_len):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, ...)
        seq_len = x.size(1)
        return self.pe[:, :seq_len].expand(x.size(0), seq_len, self.d_model)


if __name__ == '__main__':
    HIS_LENGTH = 32
    PRE_LENGTH = 256
    IN_DIM = 2
    BATCH_SIZE = 4

    model = TacGen(
        L=2,
        K=4,
        d=8,
        in_dim=IN_DIM,  # Assuming input dimension is 123
        num_his=HIS_LENGTH,
        num_pre=PRE_LENGTH,
        use_bias=True,
    )
    print('trainable parameters: {:,}'.format(model.count_parameters()))
    model.use_cuda('cuda:0')  # Move model to GPU if available

    # One Batch
    history = torch.randn(BATCH_SIZE, HIS_LENGTH, 23, IN_DIM).to('cuda:0')
    prediction = torch.randn(BATCH_SIZE, PRE_LENGTH, 23, IN_DIM).to('cuda:0')
    tgt_key_padding_mask = torch.randint(0, PRE_LENGTH-1, (BATCH_SIZE, 1)).to('cuda:0')

    src = history
    # tgt = torch.zeros_like(prediction).to('cuda:0')
    # label_pos = torch.zeros(BATCH_SIZE, PRE_LENGTH, 23, 2).to('cuda:0')
    tgt = prediction[:, :-1, :, :].to('cuda:0')
    label_pos = prediction[:, 1:, :, 0:2].to('cuda:0')

    le = torch.randn(BATCH_SIZE, 55, 768).to('cuda:0')

    for e in range(5):
        out_pos, out_stop = model(src=src, LE=le, tgt=tgt, tgt_key_padding_mask=tgt_key_padding_mask)
        label_stop = torch.zeros(BATCH_SIZE, out_stop.shape[1]).to('cuda:0')
        label_stop[torch.arange(BATCH_SIZE), tgt_key_padding_mask.squeeze(-1).long()] = 1  # (32, 128)

        time_range = torch.arange(out_stop.shape[1]).unsqueeze(0).to('cuda:0')  # (1, 128)
        mask_stop = (time_range < tgt_key_padding_mask)  # (32, 128)
        mask_pos = mask_stop.unsqueeze(2).unsqueeze(3).expand(-1, -1, 23, 2)  # Reshape to (32, 128, 23, 1)

        lc = LossCriterion(beta=1)
        l, l_mse, l_bce = lc(out_pos, label_pos, out_stop, label_stop, mask_pos, mask_stop)
        print(f'Loss: {l.item()}')

    # print(o)
    # print(o_mask)
    # print(out_pos.shape)  # Expected output shape: (32, 8, 23, 2)
    # print(out_stop)  # Expected output shape: (32, 8)
