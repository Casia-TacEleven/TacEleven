from typing import Union, Callable, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, GPT2Tokenizer, GPT2Model


class Conv2D(nn.Module):
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
        torch.nn.init.xavier_uniform_(self._conv2d.weight)

        if use_bias:
            torch.nn.init.zeros_(self._conv2d.bias)

    def forward(self, X: torch.FloatTensor) -> torch.FloatTensor:
        X = X.permute(0, 3, 2, 1)
        X = self._conv2d(X)
        if self._activation is not None:
            X = self._activation(X)
        return X.permute(0, 3, 2, 1)


class FullyConnected(nn.Module):
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
        for conv in self._conv2ds:
            X = conv(X)
        return X


class DynamicAttentionPooling(nn.Module):
    def __init__(self, embedding_dim=768):
        super(DynamicAttentionPooling, self).__init__()
        self.attention_weights = nn.Linear(embedding_dim, 1)

    def forward(self, x):
        #  x [batch_size, longest, 768]
        #  return [batch_size, 768]
        weights = F.softmax(self.attention_weights(x), dim=1)  # [batch_size, longest, 768, 1]
        pooled_output = torch.sum(x * weights, dim=1)
        return pooled_output


class SpatioTemporalEmbedding(nn.Module):
    def __init__(
        self, D: int, use_bias: bool = True
    ):
        super(SpatioTemporalEmbedding, self).__init__()
        self._fully_connected_te = FullyConnected(
            input_dims=[D, D],
            units=[D, D],
            activations=[F.relu, None],
            use_bias=use_bias,
        )
        self._fully_connected_le = FullyConnected(
            input_dims=[768, D],
            units=[D, D],
            activations=[F.relu, None],
            use_bias=use_bias,
        )
        self.layer_norm_te = nn.LayerNorm(D)
        self.layer_norm_le = nn.LayerNorm(D)
        self._dynamic_att_pool = DynamicAttentionPooling()

    def forward(
        self, TE: torch.FloatTensor, LE: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size, num_steps = TE.shape[0], TE.shape[1]
        TE = TE.unsqueeze(dim=2)
        TE = self._fully_connected_te(TE)
        TE = self.layer_norm_te(TE)

        LE = self._dynamic_att_pool(LE)
        LE = LE.unsqueeze(1).unsqueeze(1)
        LE = self._fully_connected_le(LE)
        LE = self.layer_norm_le(LE)

        TE = TE.repeat(1, 1, 23, 1)
        LE = LE.repeat(1, num_steps, 23, 1)
        return torch.concat((TE, LE), dim=-1)


class SpatialAttention(nn.Module):
    def __init__(self, K: int, d: int):
        super(SpatialAttention, self).__init__()
        D = K * d
        self._d = d
        self._K = K
        self._fully_connected_q = FullyConnected(
            input_dims=3 * D, units=D, activations=F.relu
        )
        self._fully_connected_k = FullyConnected(
            input_dims=3 * D, units=D, activations=F.relu
        )
        self._fully_connected_v = FullyConnected(
            input_dims=3 * D, units=D, activations=F.relu
        )
        self._fully_connected = FullyConnected(
            input_dims=D, units=D, activations=F.relu
        )

    def forward(
        self, X: torch.FloatTensor, STE: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
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
        X = self._fully_connected(X)
        del query, key, value, attention
        return X


class TemporalAttention(nn.Module):
    def __init__(self, K: int, d: int, mask: bool):
        super(TemporalAttention, self).__init__()
        D = K * d
        self._d = d
        self._K = K
        self._mask = mask
        self._fully_connected_q = FullyConnected(
            input_dims=3 * D, units=D, activations=F.relu
        )
        self._fully_connected_k = FullyConnected(
            input_dims=3 * D, units=D, activations=F.relu
        )
        self._fully_connected_v = FullyConnected(
            input_dims=3 * D, units=D, activations=F.relu
        )
        self._fully_connected = FullyConnected(
            input_dims=D, units=D, activations=F.relu
        )

    def forward(
        self, X: torch.FloatTensor, STE: torch.FloatTensor
    ) -> torch.FloatTensor:
        batch_size = X.shape[0]
        X = torch.cat((X, STE), dim=-1)
        query = self._fully_connected_q(X)
        key = self._fully_connected_k(X)
        value = self._fully_connected_v(X)
        query = torch.cat(torch.split(query, self._d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self._d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self._d, dim=-1), dim=0)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        attention = torch.matmul(query, key)
        attention /= self._d ** 0.5
        if self._mask:
            batch_size = X.shape[0]
            num_step = X.shape[1]
            num_nodes = X.shape[2]
            mask = torch.ones(num_step, num_step).to(X.device)
            mask = torch.tril(mask)
            mask = mask.unsqueeze(0).unsqueeze(0)
            mask = mask.repeat(self._K * batch_size, num_nodes, 1, 1)
            mask = mask.to(torch.bool)
            condition = torch.FloatTensor([-(2 ** 15) + 1]).to(X.device)
            attention = torch.where(mask, attention, condition)
        attention = F.softmax(attention, dim=-1)
        X = torch.matmul(attention, value)
        X = X.permute(0, 2, 1, 3)
        X = torch.cat(torch.split(X, batch_size, dim=0), dim=-1)
        X = self._fully_connected(X)
        del query, key, value, attention
        return X

class GatedFusion(nn.Module):
    def __init__(self, D: int):
        super(GatedFusion, self).__init__()
        self._fully_connected_xs = FullyConnected(
            input_dims=D, units=D, activations=None, use_bias=False
        )
        self._fully_connected_xt = FullyConnected(
            input_dims=D, units=D, activations=None, use_bias=True
        )
        self._fully_connected_h = FullyConnected(
            input_dims=[D, D], units=[D, D], activations=[F.relu, None],
        )

    def forward(
        self, HS: torch.FloatTensor, HT: torch.FloatTensor
    ) -> torch.FloatTensor:
        XS = self._fully_connected_xs(HS)
        XT = self._fully_connected_xt(HT)
        z = torch.sigmoid(torch.add(XS, XT))
        H = torch.add(torch.mul(z, HS), torch.mul(1 - z, HT))
        H = self._fully_connected_h(H)
        del XS, XT, z
        return H


class SpatioTemporalAttention(nn.Module):
    def __init__(self, K: int, d: int, mask: bool):
        super(SpatioTemporalAttention, self).__init__()
        self._spatial_attention = SpatialAttention(K, d)
        self._temporal_attention = TemporalAttention(K, d, mask=mask)
        self._gated_fusion = GatedFusion(K * d)

    def forward(
        self, X: torch.FloatTensor, STE: torch.FloatTensor
    ) -> torch.FloatTensor:
        HS = self._spatial_attention(X, STE)
        HT = self._temporal_attention(X, STE)
        H = self._gated_fusion(HS, HT)
        del HS, HT
        X = torch.add(X, H)
        return X


class TransformAttention(nn.Module):
    def __init__(self, K: int, d: int):
        super(TransformAttention, self).__init__()
        D = K * d
        self._K = K
        self._d = d
        self._fully_connected_q = FullyConnected(
            input_dims=2 * D, units=D, activations=F.relu
        )
        self._fully_connected_k = FullyConnected(
            input_dims=2 * D, units=D, activations=F.relu
        )
        self._fully_connected_v = FullyConnected(
            input_dims=D, units=D, activations=F.relu
        )
        self._fully_connected = FullyConnected(
            input_dims=D, units=D, activations=F.relu
        )
        self._layer_norm = nn.LayerNorm(D)

    def forward(
        self,
        X: torch.FloatTensor,
        STE_his: torch.FloatTensor,
        STE_pred: torch.FloatTensor,
    ) -> torch.FloatTensor:
        batch_size = X.shape[0]
        query = self._fully_connected_q(STE_pred)
        key = self._fully_connected_k(STE_his)
        value = self._fully_connected_v(X)
        query = torch.cat(torch.split(query, self._d, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self._d, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self._d, dim=-1), dim=0)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 3, 1)
        value = value.permute(0, 2, 1, 3)
        attention = torch.matmul(query, key)
        attention /= self._d ** 0.5
        attention = F.softmax(attention, dim=-1)
        Y = torch.matmul(attention, value)
        Y = Y.permute(0, 2, 1, 3)
        Y = torch.cat(torch.split(Y, batch_size, dim=0), dim=-1)
        Y = self._fully_connected(Y)
        del query, key, value, attention
        Y = self._layer_norm(Y)
        return Y


class TimewiseVariationalLayer(nn.Module):
    """
    Timewise Variational Layer for introducing stochasticity in temporal modeling.
    This layer applies variational encoding to each time step independently.
    """
    def __init__(self, input_dim: int, latent_dim: int = None):
        super().__init__()
        if latent_dim is None:
            latent_dim = input_dim

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder networks for mean and log variance using Linear layers
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_logvar = nn.Linear(input_dim, latent_dim)

        # Decoder network using Linear layers
        self.fc_decoder1 = nn.Linear(latent_dim, input_dim)
        self.fc_decoder2 = nn.Linear(input_dim, input_dim)

        self.layer_norm = nn.LayerNorm(input_dim)

    def reparameterize(self, mu: torch.FloatTensor, logvar: torch.FloatTensor,
                      temperature: float = 1.0, deterministic: bool = False) -> torch.FloatTensor:
        """
        Reparameterization trick for sampling from latent distribution

        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            temperature: Controls the randomness (0.0 = deterministic, 1.0 = full randomness, >1.0 = more random)
            deterministic: If True, returns mean without sampling
        """
        if deterministic:
            return mu

        std = torch.exp(0.5 * logvar) * temperature
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.FloatTensor, temperature: float = 1.0, deterministic: bool = False) -> tuple:
        """
        Forward pass through timewise variational layer
        Args:
            x: Input tensor of shape (batch_size, time_steps, nodes, features)
            temperature: Controls randomness in sampling (default 1.0)
            deterministic: If True, returns mean without sampling (default False)
        Returns:
            tuple: (reconstructed_x, mu, logvar)
        """
        batch_size, time_steps, nodes, features = x.shape

        # Vectorized approach: reshape to process all timesteps at once
        # but maintain timewise independence in the computation
        x_reshaped = x.permute(1, 0, 2, 3).contiguous()  # (time_steps, batch_size, nodes, features)
        x_reshaped = x_reshaped.view(time_steps, -1, features)  # (time_steps, batch_size*nodes, features)

        # Process all time steps in parallel
        mu_list = []
        logvar_list = []
        x_recon_list = []

        for t in range(time_steps):
            x_t = x_reshaped[t]  # (batch_size*nodes, features)

            # Encode current time step
            mu_t = self.fc_mu(x_t)  # (batch_size*nodes, latent_dim)
            logvar_t = self.fc_logvar(x_t)  # (batch_size*nodes, latent_dim)

            # Add numerical stability
            logvar_t = torch.clamp(logvar_t, min=-20, max=20)

            # Sample from latent space for current time step
            z_t = self.reparameterize(mu_t, logvar_t, temperature, deterministic)

            # Decode current time step
            x_recon_t = F.relu(self.fc_decoder1(z_t))
            x_recon_t = self.fc_decoder2(x_recon_t)
            x_recon_t = self.layer_norm(x_recon_t)

            mu_list.append(mu_t)
            logvar_list.append(logvar_t)
            x_recon_list.append(x_recon_t)

        # Stack and reshape back
        mu = torch.stack(mu_list, dim=0)  # (time_steps, batch_size*nodes, latent_dim)
        logvar = torch.stack(logvar_list, dim=0)  # (time_steps, batch_size*nodes, latent_dim)
        x_recon = torch.stack(x_recon_list, dim=0)  # (time_steps, batch_size*nodes, features)

        # Reshape back to original dimensions
        mu = mu.view(time_steps, batch_size, nodes, self.latent_dim).permute(1, 0, 2, 3)
        logvar = logvar.view(time_steps, batch_size, nodes, self.latent_dim).permute(1, 0, 2, 3)
        x_recon = x_recon.view(time_steps, batch_size, nodes, features).permute(1, 0, 2, 3)

        return x_recon, mu, logvar
    
class EncoderLayer(nn.Module):
    def __init__(self, K, d):
        super().__init__()
        D = K * d
        self._sta = SpatioTemporalAttention(K, d, mask=False)
        self._fully_connected = FullyConnected(input_dims=[D, D], units=[D, D], activations=[F.relu, None])
        self._layer_norm_1 = nn.LayerNorm(D)
        self._layer_norm_2 = nn.LayerNorm(D)

    def forward(self, X: torch.FloatTensor, STE: torch.FloatTensor) -> torch.FloatTensor:
        X = self._layer_norm_1(X + self._sta(X, STE))
        X = self._layer_norm_2(X + self._fully_connected(X))
        return X


class DecoderLayer(nn.Module):
    def __init__(self, K, d, mask=False):
        super().__init__()
        D = K * d
        self._sta = SpatioTemporalAttention(K, d, mask=mask)
        self._fully_connected = FullyConnected(input_dims=[D, D], units=[D, D], activations=[F.relu, None])
        self._layer_norm_1 = nn.LayerNorm(D)
        self._layer_norm_2 = nn.LayerNorm(D)

    def forward(self, X: torch.FloatTensor, STE_pred: torch.FloatTensor) -> torch.FloatTensor:
        X = self._layer_norm_1(X + self._sta(X, STE_pred))
        X = self._layer_norm_2(X + self._fully_connected(X))
        return X

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
        batch_size, seq_len = x.size(0), x.size(1)
        return self.pe[:, :seq_len].expand(batch_size, seq_len, self.d_model)

class TacGen(nn.Module):
    def __init__(
        self,
        L: int,
        K: int,
        d: int,
        id_dim: int = 64,
        role_id_dim: int = 32,
        pos_dim: int = 32,
        use_bias: bool=True,
        mask: str = 'false',
        use_variational: str = 'true',
    ):
        super().__init__()
        D = K * d
        if mask == 'false':
            mask = False
        if mask == 'true':
            mask = True
        if use_variational == 'false':
            use_variational = False
        if use_variational == 'true':
            use_variational = True

        self.use_variational = use_variational
        self._language_embedding = LanguageEmbedding()
        self._st_embedding = SpatioTemporalEmbedding(D, use_bias)
        self._position_encoding = PositionEncoding(d_model=D, max_len=16)
        self._encoder_layers = nn.ModuleList([EncoderLayer(K, d) for _ in range(L)])
        self._transform_attention = TransformAttention(K, d)
        print(f'the model use variational is {self.use_variational}')
        # Add variational layer
        if self.use_variational:
            self._variational_layer = TimewiseVariationalLayer(
                input_dim=D,
                latent_dim=D
            )

        self._decoder_layers = nn.ModuleList([DecoderLayer(K, d, mask=mask) for _ in range(L)])
        self._fully_connected_position = nn.Sequential(nn.Linear(2, pos_dim), nn.ReLU(), nn.LayerNorm(pos_dim))
        # self._embedding_player_id = nn.Embedding(num_embeddings=10_000_000, embedding_dim=id_dim, padding_idx=0)
        self._embedding_player_role_id = nn.Embedding(num_embeddings=50, embedding_dim=role_id_dim, padding_idx=30)
        # self.player_id_layer_norm = nn.LayerNorm(id_dim)
        self.player_role_id_layer_norm = nn.LayerNorm(role_id_dim)
        # self._fully_connected_in = FullyConnected(input_dims=[pos_dim+id_dim+role_id_dim, D], units=[D, D], activations=[F.relu, None])
        self._fully_connected_in = FullyConnected(input_dims=[pos_dim+role_id_dim, D], units=[D, D], activations=[F.relu, None])

        self.dropout = nn.Dropout(0.1)  # 添加Dropout防止过拟合
        self._fully_connected_out = FullyConnected(input_dims=[D, D], units=[D, 2], activations=[F.relu, None])

    def use_cuda(self, device):
        self.to(device)  # 将整个模型移动到 GPU
        for name, module in self.named_children():
            module.to(device)  # 遍历所有子模块并移动到 GPU
        self._language_embedding.use_cuda(device)

    def forward(
        self, X: torch.FloatTensor, TE_x: torch.FloatTensor, TE_y: torch.FloatTensor, lang_list: list,
        temperature: float = 1.0, deterministic: bool = False
    ) -> torch.FloatTensor:
        LE = self._language_embedding.embed(lang_list)  # (batch_size, num_steps, 768)

        position = X[..., :2]
        player_id = X[..., 2].long()
        player_role_id = X[..., 3].long()
        player_role_id = torch.where(player_role_id == -1, 30, player_role_id)

        # 分别处理并标准化各特征，确保scale一致
        pos_feat = self._fully_connected_position(position)
        # id_feat = self.player_id_layer_norm(self._embedding_player_id(player_id))
        role_feat = self.player_role_id_layer_norm(self._embedding_player_role_id(player_role_id))

        # X = self.dropout(self._fully_connected_in(torch.concat([pos_feat, id_feat, role_feat], dim=-1)))  # -> D dim
        X = self.dropout(self._fully_connected_in(torch.concat([pos_feat, role_feat], dim=-1)))  # -> D dim

        TE_x = self._position_encoding(TE_x)
        TE_y = self._position_encoding(TE_y)
        STE_his = self._st_embedding(TE_x, LE)
        STE_pred = self._st_embedding(TE_y, LE)

        # Encoder layers
        for net in self._encoder_layers:
            X = net(X, STE_his)

        # Transform attention (cross-attention)
        H = self._transform_attention(X, STE_his, STE_pred)

        # Apply variational layer if enabled
        if self.use_variational:
            H_var, mu, logvar = self._variational_layer(H, temperature, deterministic)
            H = H_var  # Use variational output
        else:
            pass

        # Decoder layers
        for net in self._decoder_layers:
            H = net(H, STE_pred)

        Y = torch.squeeze(self._fully_connected_out(H), 3)

        if self.use_variational:
            return Y, mu, logvar
        else:
            return Y

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LossCriterion(nn.Module):
    def __init__(self, kl_weight=1e-4):
        super().__init__()
        self.kl_weight = kl_weight

    def kl_divergence(self, mu, logvar):
        """
        Calculate KL divergence for variational loss with numerical stability
        KL(q(z|x) || p(z)) where p(z) = N(0, I)
        """
        # Clamp logvar to prevent extreme values
        logvar = torch.clamp(logvar, min=-20, max=20)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        # Use mean to reduce the tensor and improve stability
        return kl.mean()

    def forward(self, output, label, mu=None, logvar=None):
        if isinstance(output, tuple):
            # If model returns tuple (for variational version)
            if len(output) == 3:
                Y_hat, mu, logvar = output
            else:
                Y_hat = output[0]
                mu, logvar = output[1], output[2] if len(output) > 2 else (None, None)
        else:
            Y_hat = output

        # Reconstruction loss
        recon_loss = F.mse_loss(Y_hat[..., :2], label[..., :2])

        # KL divergence loss (if variational)
        if mu is not None and logvar is not None:
            kl_loss = self.kl_divergence(mu, logvar)
            total_loss = recon_loss + self.kl_weight * kl_loss
            return total_loss, recon_loss, kl_loss
        else:
            return recon_loss


class LanguageEmbedding:
    def __init__(
        self,
        device: str = 'cpu',
        model_name: str = '/home/trl/trl/hf_hub/models/google-bert-bert-base-uncased'
    ):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(device)
        self.device = device

    def use_cuda(self, device: str):
        self.model.to(device)
        self.device = device

    def embed(self, texts: list, max_length: int = 256) -> torch.FloatTensor:
        with torch.no_grad():
            encoded = self.tokenizer(texts, padding='longest', truncation=True, return_tensors='pt')
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            json_special_tokens = ['{', '}', '[', ']', ':', ',']
            json_token_ids = self.tokenizer.convert_tokens_to_ids(json_special_tokens)
            json_mask = torch.zeros_like(input_ids, dtype=torch.int).to(self.device)
            for token_id in json_token_ids:
                json_mask = json_mask + (input_ids == token_id).int()
            combined_mask = attention_mask * (1 - json_mask)
            combined_mask = attention_mask
            outputs = self.model(input_ids=input_ids, attention_mask=combined_mask, output_hidden_states=True)
            last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, sequence_length, hidden_size)
        return last_hidden_state  # pooler_output=tensor[batch_size, max_length, 768]


if __name__ == '__main__':
    # Test standard model
    print("Testing standard model:")
    model = TacGen(
        L=4,
        K=4,
        d=256,
        use_variational=False
    )
    model.eval()
    # x = torch.randn(32, 3, 23, 2)
    x = torch.ones(32, 3, 23, 2)
    x_2 = torch.ones_like(x)
    x = torch.concat([x, x_2],dim=-1)
    te_x = torch.ones(32, 3, 3)
    te_y = torch.ones(32, 5, 3)
    le = ['a']*32

    output = model(X=x, TE_x=te_x, TE_y=te_y, lang_list=le)
    print(f"{output[0,0,0,:]}")
    output = model(X=x, TE_x=te_x, TE_y=te_y, lang_list=le)
    print(f"{output[0,0,0,:]}")
    output = model(X=x, TE_x=te_x, TE_y=te_y, lang_list=le)
    print(f"{output[0,0,0,:]}")

    # Test variational model with different modes
    print("\nTesting variational model:")
    model_var = TacGen(
        L=4,
        K=4,
        d=256,
        use_variational=True,
    )

    # Mode 1: Deterministic (traditional VAE inference)
    print("\n1. Deterministic mode (traditional VAE):")
    output_det = model_var(X=x, TE_x=te_x, TE_y=te_y, lang_list=le, deterministic=True)
    if isinstance(output_det, tuple):
        Y_det, mu, logvar = output_det
        print(f"{Y_det[0,0,0,:]}")
    output_det = model_var(X=x, TE_x=te_x, TE_y=te_y, lang_list=le, deterministic=True)
    if isinstance(output_det, tuple):
        Y_det, mu, logvar = output_det
        print(f"{Y_det[0,0,0,:]}")
    output_det = model_var(X=x, TE_x=te_x, TE_y=te_y, lang_list=le, deterministic=True)
    if isinstance(output_det, tuple):
        Y_det, mu, logvar = output_det
        print(f"{Y_det[0,0,0,:]}")
        
    # Mode 2: Stochastic with standard temperature
    print("\n2. Stochastic mode (standard temperature=1.0):")
    output_stoch = model_var(X=x, TE_x=te_x, TE_y=te_y, lang_list=le, deterministic=False)
    if isinstance(output_stoch, tuple):
        Y_det, mu, logvar = output_stoch
        print(f"{Y_det[0,0,0,:]}")
    output_stoch = model_var(X=x, TE_x=te_x, TE_y=te_y, lang_list=le, deterministic=False)
    if isinstance(output_stoch, tuple):
        Y_det, mu, logvar = output_stoch
        print(f"{Y_det[0,0,0,:]}")
    output_stoch = model_var(X=x, TE_x=te_x, TE_y=te_y, lang_list=le, deterministic=False)
    if isinstance(output_stoch, tuple):
        Y_det, mu, logvar = output_stoch
        print(f"{Y_det[0,0,0,:]}")
        
    # Mode 3: High diversity (higher temperature)
    print("\n3. High diversity mode (temperature=1.5):")
    output_high = model_var(X=x, TE_x=te_x, TE_y=te_y, lang_list=le, temperature=1.5)
    if isinstance(output_high, tuple):
        Y_high, mu, logvar = output_high
        print(f"   Output shape: {Y_high.shape}")

    # Mode 4: Low diversity (lower temperature)
    print("\n4. Low diversity mode (temperature=0.5):")
    output_low = model_var(X=x, TE_x=te_x, TE_y=te_y, lang_list=le, temperature=0.5)
    if isinstance(output_low, tuple):
        Y_low, mu, logvar = output_low
        print(f"   Output shape: {Y_low.shape}")

    # Demonstrate diversity by multiple generations
    print("\n5. Diversity demonstration (3 different outputs with same input):")
    for i in range(3):
        output_diverse = model_var(X=x, TE_x=te_x, TE_y=te_y, lang_list=le, temperature=1.0)
        if isinstance(output_diverse, tuple):
            Y_diverse, _, _ = output_diverse
            print(f"   Generation {i+1} - Mean position: {Y_diverse[0, 0, 0].detach().numpy()}")

    # Test loss criterion
    print("\nTesting loss criterion:")
    criterion = LossCriterion(kl_weight=1e-4)
    target = torch.randn(32, 5, 23, 2)

    # Standard loss
    loss_std = criterion(output, target)
    print(f"Standard loss: {loss_std.item():.4f}")

    # Variational loss
    loss_result = criterion(output_stoch, target)
    if isinstance(loss_result, tuple):
        total_loss, recon_loss, kl_loss = loss_result
        print(f"Variational total loss: {total_loss.item():.4f}")
        print(f"Reconstruction loss: {recon_loss.item():.4f}")
        print(f"KL loss: {kl_loss.item():.4f}")

    print(f"\nModel parameters: {model_var.count_parameters():,}")

    print("\n" + "="*60)
    print("USAGE RECOMMENDATIONS:")
    print("="*60)
    print("🎯 For diverse tactical generation: temperature=1.0-1.5")
    print("🎯 For consistent predictions: temperature=0.3-0.7")
    print("🎯 For deterministic mode: deterministic=True")
    print("🎯 For training: always use temperature=1.0")
