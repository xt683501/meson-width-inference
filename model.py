import torch
from torch import nn, einsum
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from einops import rearrange, repeat

from hyper_connections import HyperConnections

# feedforward and attention

class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

def FeedForward(dim, mult = 4, dropout = 0.):
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, dim * mult * 2),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(dim * mult, dim)
    )

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads

        x = self.norm(x)

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        attn = sim.softmax(dim = -1)
        dropped_attn = self.dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', dropped_attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        out = self.to_out(out)

        return out, attn

# transformer

class Transformer(Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        attn_dropout,
        ff_dropout,
        num_residual_streams = 4
    ):
        super().__init__()

        init_hyper_conn, self.expand_streams, self.reduce_streams = HyperConnections.get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                init_hyper_conn(dim = dim, branch = Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout)),
                init_hyper_conn(dim = dim, branch = FeedForward(dim, dropout = ff_dropout)),
            ]))

    def forward(self, x, return_attn = False):
        post_softmax_attns = []

        x = self.expand_streams(x)

        for attn, ff in self.layers:
            x, post_softmax_attn = attn(x)
            post_softmax_attns.append(post_softmax_attn)

            x = ff(x)

        x = self.reduce_streams(x)

        if not return_attn:
            return x

        return x, torch.stack(post_softmax_attns)

# numerical embedder

class NumericalEmbedder(Module):
    def __init__(self, dim, num_numerical_types):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_numerical_types, dim))
        self.biases = nn.Parameter(torch.randn(num_numerical_types, dim))

    def forward(self, x):
        x = rearrange(x, 'b n -> b n 1')
        return x * self.weights + self.biases

# main class

class FTTransformer(Module):
    def __init__(
        self,
        *,
        # --- 配置 ---
        categorical_cols: list,      # e.g., ['P', 'C', 'G']
        categorical_cardinalities: list, # e.g., [3, 3, 3] (每个类别特征有多少个唯一值)
        combinable_numerical_cols: list, # e.g., ['N', 'I', 'J', 'I3']
        is_known_col_map: dict,          # e.g., {'N': 'Nk', 'I': 'Ik', ...}
        quark_cols: list,                # e.g., ['u', 'ubar', 'd', ...]
        # --- 原始参数 ---
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        #num_special_tokens = 2,
        attn_dropout = 0.,
        ff_dropout = 0.,
        num_residual_streams = 4
    ):
        super().__init__()
        #assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        #assert len(categories) + num_continuous > 0, 'input shape must not be null'

        # categorical embedding

        # --- 类别特征 ---
        self.categorical_cols = categorical_cols
        self.num_categories = len(self.categorical_cols)

        if self.num_categories > 0:
            # 这里的 categorical_cardinalities 就是配置的 [3, 3, 3] 之类的列表
            self.num_unique_categories = sum(categorical_cardinalities)
            # 废除 special_tokens
            total_tokens = self.num_unique_categories

            # 计算偏移量的逻辑不变
            # 注意：因为废除了 special tokens，所以 pad 的 value 是 0
            categories_offset = F.pad(torch.tensor(list(categorical_cardinalities)), (1, 0), value=0)
            categories_offset = categories_offset.cumsum(dim=-1)[:-1]
            self.register_buffer('categories_offset', categories_offset)

            # 创建类别嵌入层
            self.categorical_embeds = nn.Embedding(total_tokens, dim)


        # continuous

        # --- 数值特征 ---
        self.combinable_numerical_cols = combinable_numerical_cols
        self.is_known_col_map = is_known_col_map
        self.quark_cols = quark_cols

        # 为需要组合的特征创建嵌入器 (_raw 和 _is_known)
        # 它的输入是 len(...) * 2 列数据
        num_combinable_inputs = len(self.combinable_numerical_cols) * 2
        if num_combinable_inputs > 0:
            self.combinable_embedder = NumericalEmbedder(dim, num_combinable_inputs)

        # 为夸克特征创建独立的嵌入器
        num_quark_inputs = len(self.quark_cols)
        if num_quark_inputs > 0:
            self.quark_embedder = NumericalEmbedder(dim, num_quark_inputs)

        # 根据列名创建专属的混合器
        self.combination_layers = nn.ModuleDict()
        for feat_name in self.combinable_numerical_cols:
            self.combination_layers[feat_name] = nn.Linear(dim * 2, dim)

        # 为每个可组合特征创建专属的“未知”状态嵌入向量
        # 使用 nn.ParameterDict 来管理这些可学习的向量
        self.unknown_embeddings = nn.ParameterDict()
        for feat_name in self.combinable_numerical_cols:
            # 为每个特征创建一个维度为 dim 的、可学习的“未知”状态向量
            # nn.Parameter 会将这个张量注册为模型的可训练参数
            self.unknown_embeddings[feat_name] = nn.Parameter(torch.randn(dim))

        # cls token

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # transformer

        self.transformer = Transformer(
            dim = dim,
            depth = depth,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            num_residual_streams = num_residual_streams
        )

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
          # nn.ReLU(),综合考虑还是不要这个激活函数了，会导致系统性偏差
            nn.Linear(dim, dim_out)
        )

    #  FTTransformer 类里面的方法
    def forward(self, x: dict, return_attn=False):
        """
        模型的前向传播方法。
        :param x: 一个字典，键是特征的列名(str)，值是对应的PyTorch张量(shape: [B])。
        :param return_attn: 是否返回注意力权重。
        :return: 模型的输出 logits，以及可选的注意力权重。
        """
        # --- 0. 初始化内部变量 ---

        x_categ_data = None
        x_combinable_data = None
        x_quark_data = None

        # --- 1. 按配置名，从输入字典 x 中分拣出各类特征的原始数据 ---

        # 分拣出类别特征张量
        if self.num_categories > 0:
            # 使用列表推导式高效地收集所有类别特征的张量
            categ_tensors = [x[col] for col in self.categorical_cols]
            # 使用 torch.stack 将列表堆叠成一个 [B, num_categories] 的张量
            x_categ_data = torch.stack(categ_tensors, dim=1).long()

        # 分拣出需要组合的数值特征 (_raw 和 _is_known)
        if len(self.combinable_numerical_cols) > 0:
            combinable_tensors = []
            # 严格按照 self.combinable_numerical_cols 的顺序来保证一致性
            for feat_name in self.combinable_numerical_cols:
                raw_col_name = feat_name
                known_col_name = self.is_known_col_map[feat_name]
                combinable_tensors.append(x[raw_col_name])
                combinable_tensors.append(x[known_col_name])
            x_combinable_data = torch.stack(combinable_tensors, dim=1).float()

        # 分拣出夸克特征
        if len(self.quark_cols) > 0:
            quark_tensors = [x[col] for col in self.quark_cols]
            x_quark_data = torch.stack(quark_tensors, dim=1).float()

        # --- 2. 将分拣好的数据送入各自的嵌入器并处理 ---
        all_embeddings = [] # 我们将把所有最终的特征嵌入向量收集到这里

        # 处理类别特征
        if x_categ_data is not None:
            # 加上偏移量，送入嵌入层
            x_categ_emb = self.categorical_embeds(x_categ_data + self.categories_offset)
            all_embeddings.append(x_categ_emb)

        # 处理夸克特征
        if x_quark_data is not None:
            emb_quarks = self.quark_embedder(x_quark_data)
            all_embeddings.append(emb_quarks)

        # --- 3. 【核心】处理组合特征 ---
        if x_combinable_data is not None:
            # 先把所有组合特征的原始数据送入嵌入器
            emb_combinable_pairs = self.combinable_embedder(x_combinable_data) # shape: [B, num_pairs*2, D]

            # 然后逐对处理，应用混合
            final_combined_embeddings = []
            for i, feat_name in enumerate(self.combinable_numerical_cols):
                # 取出属于当前特征对的 raw 和 known 嵌入向量
                raw_emb = emb_combinable_pairs[:, i*2]
                known_emb = emb_combinable_pairs[:, i*2 + 1]

                # --- 可学习未知嵌入开关】 ---
                # 1. 找到哪些样本的当前特征是未知的
                is_unknown_mask = (x_combinable_data[:, i * 2 + 1] == 0)  # shape: [B]

                # 2. 从我们创建的 ParameterDict 中，获取当前特征专属的“未知”嵌入向量
                unknown_emb_for_this_feat = self.unknown_embeddings[feat_name]  # shape: [D]

                # 3. 使用 torch.where 实现条件替换
                #    - 条件: is_unknown_mask.unsqueeze(1) -> 变形为 [B, 1] 以便广播
                #    - 如果条件为 True (未知)，则使用 unknown_emb_for_this_feat
                #    - 如果条件为 False (已知)，则保持原始的 raw_emb
                raw_emb = torch.where(
                    is_unknown_mask.unsqueeze(1),
                    unknown_emb_for_this_feat,  # PyTorch会自动将[D]广播到[B, D]
                    raw_emb
                )

                # 送入专属的混合器(combination_layers)
                combined = torch.cat([raw_emb, known_emb], dim=-1)
                final_emb = self.combination_layers[feat_name](combined) # shape: [B, D]
                final_combined_embeddings.append(final_emb)

            # 将处理好的几个组合特征堆叠起来
            x_combined_emb = torch.stack(final_combined_embeddings, dim=1) # shape: [B, num_pairs, D]
            all_embeddings.append(x_combined_emb)

        # --- 4. 最终组装与调用 ---

        # 按照 类别 -> 夸克 -> 组合数值 的顺序拼接所有令牌
        x = torch.cat(all_embeddings, dim=1)

        # Prepend CLS token
        b = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        # 送入Transformer (现在掩码机制已经废除)
        x, attns = self.transformer(x, return_attn=True)

        # 提取CLS令牌的输出
        x = x[:, 0]

        # 通过输出头得到最终预测值
        logits = self.to_logits(x)

        # 返回结果
        if not return_attn:
            return logits
        return logits, attns
