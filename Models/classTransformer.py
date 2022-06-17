import math
import torch
import copy
import warnings
from torch.overrides import (
    has_torch_function,
    handle_torch_function)
from typing import Optional, Tuple

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import _LinearWithBias
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter
from Models.classSimpleModel import CNN1DDoubleMaxpool

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def multi_head_attention_forward(query: Tensor, key: Tensor, value: Tensor, d_model_to_check: int, num_heads: int,
                                 in_proj_weight: Tensor, in_proj_bias: Tensor, bias_k: Optional[Tensor],
                                 bias_v: Optional[Tensor],
                                 add_zero_attn: bool, dropout_p: float, out_proj_weight: Tensor, out_proj_bias: Tensor,
                                 training: bool = True, key_padding_mask: Optional[Tensor] = None,
                                 need_weights: bool = True,
                                 attn_mask: Optional[Tensor] = None, use_separate_proj_weight: bool = False,
                                 q_proj_weight: Optional[Tensor] = None,
                                 k_proj_weight: Optional[Tensor] = None,
                                 v_proj_weight: Optional[Tensor] = None,
                                 static_k: Optional[Tensor] = None,
                                 static_v: Optional[Tensor] = None):
    """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        d_model_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    # print('---- calcul attention :')
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(multi_head_attention_forward, tens_ops, query, key, value, d_model_to_check,
                                     num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn, dropout_p,
                                     out_proj_weight, out_proj_bias, training=training,
                                     key_padding_mask=key_padding_mask, need_weights=need_weights,
                                     attn_mask=attn_mask, use_separate_proj_weight=use_separate_proj_weight,
                                     q_proj_weight=q_proj_weight,
                                     k_proj_weight=k_proj_weight,
                                     v_proj_weight=v_proj_weight,
                                     static_k=static_k,
                                     static_v=static_v)
    tgt_len, bsz, d_model = query.size()
    assert d_model == d_model_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    # print('- def dim head :')
    head_dim = d_model // num_heads
    assert head_dim * num_heads == d_model, "d_model must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    # print('- def F.linear for q, k, v :')
    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or torch.equal(key, value)):
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

            # print(' !!!on verifie taille de q : {}'.format(q.shape))
        elif key is value or torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = d_model
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = d_model
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = d_model
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = d_model
            _end = d_model * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = d_model * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == d_model and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == d_model and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == d_model and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:d_model])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[d_model: (d_model * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(d_model * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    # print('- verif attn_mask :')
    if attn_mask is not None:
        assert (
                attn_mask.dtype == torch.float32
                or attn_mask.dtype == torch.float64
                or attn_mask.dtype == torch.float16
                or attn_mask.dtype == torch.uint8
                or attn_mask.dtype == torch.bool
        ), "Only float, byte, and bool types are supported for attn_mask, not {}".format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            # print('a l interieur attentions : attn mash size = {}'.format(attn_mask.size()))
            # print('compare to {}'.format([1, query.size(0), key.size(0)]))
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 2D attn_mask is not correct.")
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError("The size of the 3D attn_mask is not correct.")
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # print('- verif key_padding_mask :')
    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead."
        )
        key_padding_mask = key_padding_mask.to(torch.bool)

    # print('- gestion bias :')
    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    # print('- reshape q, k, v :')
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    # print(' !!!on verifie taille de q : {}'.format(q.shape))
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # print(' !!!on verifie taille de k : {}'.format(k.shape))
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        # print(' !!!on verifie taille de v : {}'.format(v.shape))

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # print('- calcul attention q, k :')
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    # print(' !!!on verifie taille de attn_output_weights : {}'.format(attn_output_weights.shape))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        # print('- apply attn_mask :')
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        else:
            attn_output_weights += attn_mask

    if key_padding_mask is not None:
        # print('- apply key_padding_mask :')
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float("-inf"),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    # print('- add softmax, dropout :')
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    # print('- calcul attention + v :')
    # print(' !!!on verifie taille de attn_output_weights : {}'.format(attn_output_weights.shape))
    # print(' !!!on verifie taille de v : {}'.format(v.shape))
    attn_output = torch.bmm(attn_output_weights, v)
    # print(' !!!on verifie taille de attn_output : {}'.format(attn_output.shape))
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, d_model)
    # print(' !!!on verifie taille de attn_output : {}'.format(attn_output.shape))
    # print(' !!!on verifie taille de out_proj_weight : {}'.format(out_proj_weight.shape))
    # print(' !!!on verifie taille de out_proj_bias : {}'.format(out_proj_bias.shape))
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)
    # print(' !!!on verifie taille de attn_output : {}'.format(attn_output.shape))

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None



class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        d_model: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, d_model, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.kdim = kdim if kdim is not None else d_model
        self.vdim = vdim if vdim is not None else d_model
        self._qkv_same_embed_dim = self.kdim == d_model and self.vdim == d_model

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = d_model // num_heads
        assert self.head_dim * num_heads == self.d_model, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(d_model, d_model))
            self.k_proj_weight = Parameter(torch.Tensor(d_model, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(d_model, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * d_model, d_model))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * d_model))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(d_model, d_model)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, d_model))
            self.bias_v = Parameter(torch.empty(1, 1, d_model))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.

          If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
          the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

    Shapes for outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if not self._qkv_same_embed_dim:
            return multi_head_attention_forward(
                query, key, value, self.d_model, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            # print('---- MultiHeadAttention ----')
            # print('k size = {}'.format(key.shape))
            # print('q size = {}'.format(query.shape))
            # print('v size = {}'.format(value.shape))
            # print('mask size = {}'.format(attn_mask.shape))
            # print('mask size post unsqueeze = {}'.format(attn_mask.unsqueeze(0).shape))
            return multi_head_attention_forward(
                query, key, value, self.d_model, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


class EncoderLayer(nn.Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = nn.EncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        # print('on def class TransformerEncoderLayer avec d_model = {}'.format(d_model))
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout=dropout)

        residual_layers = ResidualConnection(d_model, dropout)
        self.residual = _get_clones(residual_layers, 2)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(EncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # print('---- EncoderLayer ----')
        # Attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        # print('dans encoder layers source size = {} et weights size = {}'.format(src2.shape, attn_output_weight.shape))
        src = self.residual[0](src, src2)

        # FeedForward
        src2 = self.feed_forward(src)
        src = self.residual[1](src, src2)
        # print('----------------------')
        return src


class Encoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.EncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.Encoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # print('---- TransformerEncoder ----')
        # print('source size = {}'.format(src.shape))
        # print('msk size = {}'.format(mask.shape))
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        # print('post encoder layers = {}'.format(output.shape))

        if self.norm is not None:
            output = self.norm(output)
        # print('post norm layers = {}'.format(output.shape))
        # print('---------------------------')
        return output


class DecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.DecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.enc_dec_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout=dropout)

        residual_layers = ResidualConnection(d_model, dropout)
        self.residual = _get_clones(residual_layers, 3)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(DecoderLayer, self).__setstate__(state)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # self Attention
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = self.residual[0](tgt, tgt2)

        # Encoder Decoder Attention
        tgt2 = self.enc_dec_attn(tgt, memory, memory, attn_mask=memory_mask,
                                 key_padding_mask=memory_key_padding_mask)[0]
        tgt = self.residual[1](tgt, tgt2)

        # FeedForward
        tgt2 = self.feed_forward(tgt)
        tgt = self.residual[2](tgt, tgt2)
        return tgt


class Decoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.DecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.Decoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(Decoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class PositionalEncoding(nn.Module):
    """
    Classe qui fait le positional encoding.

    Attributes:
        dropout (float) : dropout
        register_buffer : ?
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        The constructor for PositionalEncoding.

        Parameters:
            d_model (int): dim du model tout au long du Transformer
            dropout (float): dropout
            max_len (int) : taille max de la source

        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Conv1dEmbedding(nn.Module):
    def __init__(self, channel_input, l_input, layers_param):
        super(Conv1dEmbedding, self).__init__()

        self.cnn_1 = CNN1DDoubleMaxpool(channel_input, l_input,
                                        layers_param['cnn1'],
                                        layers_param['kernel_1'],
                                        layers_param['maxpool_1'],
                                        self.dropout,
                                        self.batch_norm)
        self.cnn_2 = CNN1DDoubleMaxpool(self.cnn_1.output_channel, self.cnn_1.l_out,
                                        layers_param['cnn2'],
                                        layers_param['kernel_2'],
                                        layers_param['maxpool_2'],
                                        self.dropout,
                                        self.batch_norm)
        self.cnn_3 = CNN1DDoubleMaxpool(self.cnn_2.output_channel, self.cnn_2.l_out,
                                        layers_param['cnn3'],
                                        layers_param['kernel_3'],
                                        layers_param['maxpool_3'],
                                        self.dropout,
                                        self.batch_norm)

    def forward(self, x):
        # print('x size = {}'.format(x.size()))
        #met batch dim en 0
        c_in = torch.transpose(x, 0, 1)
        # print(c_in.size())
        # met src size en last
        c_in = torch.transpose(c_in, 1, -1)
        # print(c_in.size())
        c_out = self.cnn_1(c_in)
        # print(c_out.size())
        c_out = self.cnn_2(c_out)
        # print(c_out.size())
        c_out = self.cnn_3(c_out)
        # print(c_out.size())
        c_out = torch.transpose(c_out, 1, -1)
        # print(c_out.size())
        c_out = torch.transpose(c_out, 0, 1)
        # print(c_out.size())

        return c_out


class Embedding(nn.Module):
    def __init__(self, type, d_model, vocab, layers_param, src_size):
        super(Embedding, self).__init__()

        if type == 'nn':
            self.lut = nn.Embedding(vocab, d_model)
            self.init_weights()
        elif type == 'lin':
            self.lut = nn.Linear(1, d_model)
            self.init_weights()
        else:
            self.lut = Conv1dEmbedding(1, src_size, layers_param)
        self.d_model = d_model

    def init_weights(self):
        initrange = 0.1
        self.lut.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class FeedForward(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(FeedForward, self).__init__()

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(FeedForward, self).__setstate__(state)

    def forward(self, src: Tensor) -> Tensor:
        src = self.activation(self.linear1(src))
        src = self.dropout(src)
        src = self.linear2(src)
        return src


class ResidualConnection(nn.Module):
    """
    A residual connection followed by a layer norm.

    """

    def __init__(self, d_model, dropout):
        super(ResidualConnection, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_prev):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(x_prev))


class TransformerModel(nn.Module):
    """
        Classe qui cree le Transformer.

    Attributes:
        model_type (str): type du model
        pos_encoder (class): positional encoding
        transformer_encoders (class): couche encoder
        embedding_input (class): embedding class
        d_model (int): embedding dimension
        transformer_decoder (class): couche decoder
    """

    def __init__(self, src_ntoken, tgt_ntoken, criterion_type, layers_param, src_size, output_size):
        """
        The constructor for TransformerModel.

        Parameters:
              src_len (int): size of input vocabulary
              tgt_ntoken (int): size of output vocabulary
              d_model (int): inside model dimension
              nhead (int): the number of heads in the multiheadattention models
              nlayers (int): the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
              dropout (float): the dropout value

        """
        super(TransformerModel, self).__init__()

        self.d_model = layers_param['d_model']
        self.m_type = 'transformer'
        self.embedding_type = layers_param['embedding_type']

        # embedding !!! dans papier parle d'un dropout qui n'est pas ici !!!
        self.embedding_input = Embedding(layers_param['embedding_type'], layers_param['d_model'], src_ntoken, layers_param, src_size)
        if tgt_ntoken is not None:
            self.embedding_output = Embedding(layers_param['embedding_type'], layers_param['d_model'], tgt_ntoken, layers_param, src_size)

        # pos encoding
        self.pos_encoder = PositionalEncoding(layers_param['d_model'], layers_param['dropout'])

        # encoder
        encoder_layers = EncoderLayer(layers_param['d_model'], layers_param['nhead'], layers_param['dim_feedforward'],
                                      layers_param['dropout'])
        self.transformer_encoder = Encoder(encoder_layers, layers_param['nlayers'])

        # decoder
        if tgt_ntoken is not None:
            decoder_layers = DecoderLayer(layers_param['d_model'], layers_param['nhead'],
                                          layers_param['dim_feedforward'], layers_param['dropout'])
            self.transformer_decoder = Decoder(decoder_layers, layers_param['nlayers'])
        #     self.out = nn.Linear(layers_param['d_model'], tgt_ntoken if output_size is None else output_size)
        # else:
        #     self.out = nn.Linear(src_size*layers_param['d_model'], src_ntoken if output_size is None else output_size)
        self.generator = Generator(src_ntoken, tgt_ntoken, criterion_type, layers_param, src_size, output_size)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):

        # embedding et pos encoding
        src = self.embedding_input(src) #* math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if tgt is not None:
            tgt = self.embedding_output(tgt) #* math.sqrt(self.d_model)
            tgt = self.pos_encoder(tgt)

        # encoder
        src = self.transformer_encoder(src, src_mask)

        # decoder
        if tgt is not None:
            output = self.transformer_decoder(tgt, src, tgt_mask, src_mask)
            # output = self.out(output)
        else:
            output = src
            # src = torch.transpose(src, 0, 1).contiguous().view(-1, self.src_size * self.d_model)
            # output = self.out(src)

        return output


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, src_ntoken, tgt_ntoken, criterion_type, layers_param, src_size, output_size):
        super(Generator, self).__init__()

        self.d_model = layers_param['d_model']
        self.criterion_type = criterion_type
        self.src_size = src_size
        self.output_size = output_size
        self.src_ntoken = src_ntoken
        self.tgt_ntoken = tgt_ntoken
        self.m_type = 'transformer'

        if output_size == 1 or output_size == src_ntoken or output_size == tgt_ntoken:
            self.proj = nn.Linear(layers_param['d_model'], output_size)
        else:
            self.proj = nn.Linear(src_size * layers_param['d_model'], output_size)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1

        self.proj.bias.data.zero_()
        self.proj.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):

        # print(x.shape)
        if self.criterion_type == 'cross_entropy_loss':
            if self.output_size != 1 and self.output_size != self.src_ntoken and self.output_size != self.tgt_ntoken:
                # print(x.shape)
                x = torch.transpose(x, 0, 1)
                # print(x.shape)
                x = x.contiguous().view(-1, self.src_size * self.d_model)
            output = self.proj(x)
        else:
            output = self.proj(x)
            output = F.log_softmax(output, dim=-1)

        return output


class TransformerModelRef(nn.Module):
    """
        Classe qui cree le Transformer.

    Attributes:
        model_type (str): type du model
        pos_encoder (class): positional encoding
        transformer_encoders (class): couche encoder
        embedding_input (class): embedding class
        d_model (int): embedding dimension
        transformer_decoder (class): couche decoder
    """

    def __init__(self, src_ntoken, tgt_ntoken, src_size, d_model, dim_feedforward, nhead, nlayers, dropout, embedding_type):
        """
        The constructor for TransformerModel.

        Parameters:
              src_len (int): size of input vocabulary
              tgt_ntoken (int): size of output vocabulary
              d_model (int): inside model dimension
              nhead (int): the number of heads in the multiheadattention models
              nlayers (int): the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
              dropout (float): the dropout value

        """
        super(TransformerModelRef, self).__init__()
        self.m_type = 'transformer'
        self.d_model = d_model
        self.src_size = src_size
        self.embedding_type = 'nn'

        if embedding_type == 'nn':
            self.embedding_input = nn.Embedding(src_ntoken, d_model)
            if tgt_ntoken is not None:
                self.embedding_output = nn.Embedding(tgt_ntoken, d_model)
        else:
            self.embedding_input = nn.Linear(1, d_model)
            if tgt_ntoken is not None:
                self.embedding_output = nn.Linear(1, d_model)

        # pos encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # encoder
        encoder_layers = EncoderLayer(d_model, nhead,
                                      dim_feedforward, dropout)
        self.transformer_encoder = Encoder(encoder_layers, nlayers)

        # decoder
        if tgt_ntoken is not None:
            decoder_layers = DecoderLayer(d_model, nhead,
                                          dim_feedforward, dropout)
            self.transformer_decoder = Decoder(decoder_layers, nlayers)
            self.out = nn.Linear(d_model, tgt_ntoken)
        else:
            self.out = nn.Linear(src_size*d_model, src_ntoken)

        self.init_weights(tgt_ntoken)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self, tgt_ntoken):
        initrange = 0.1
        self.embedding_input.weight.data.uniform_(-initrange, initrange)
        if tgt_ntoken is not None:
            self.embedding_output.weight.data.uniform_(-initrange, initrange)

        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None):

        # embedding et pos encoding
        src = self.embedding_input(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if tgt is not None:
            tgt = self.embedding_input(tgt) * math.sqrt(self.d_model)
            tgt = self.pos_encoder(tgt)

        # encoder
        src = self.transformer_encoder(src, src_mask)

        # decoder
        if tgt is not None:
            output = self.transformer_decoder(tgt, src, tgt_mask, src_mask)
            output = self.out(output)
        else:
            src = torch.transpose(src, 0, 1).contiguous().view(-1, self.src_size*self.d_model)
            output = self.out(src)

        return output
