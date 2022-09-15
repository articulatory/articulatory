"""Misc PyTorch Layers

Transformer from Gaddy and Klein, 2021, https://arxiv.org/pdf/2106.01933.pdf 
    Original code:
        https://github.com/dgaddy/silent_speech/blob/master/transformer.py
"""

import logging
import torch
import torch.nn.functional as F

from torch import nn
from torch.nn.utils import weight_norm
from typing import Optional


def WNConv1d(*args, **kwargs):
    '''
    https://github.com/descriptinc/cargan/blob/master/cargan/model/gantts/generator.py 
    '''
    return weight_norm(torch.nn.Conv1d(*args, **kwargs))


def get_padding(kernel_size, dilation=1):
    '''Computes padding for non-strided dilated convolution
    
    https://github.com/descriptinc/cargan/blob/master/cargan/model/gantts/generator.py 
    '''
    return dilation * (kernel_size - 1) // 2


class GBlock(torch.nn.Module):
    '''
    https://github.com/descriptinc/cargan/blob/master/cargan/model/gantts/generator.py 
    '''
    def __init__(self, input_dim, output_dim, upsample=1, kernel_size=3, extra_art=False, norm=True):
        '''
        Assumes kernel_size is odd i think
        '''
        super().__init__()

        conv_type = WNConv1d if norm else torch.nn.Conv1d

        ############################################################
        # Create first residual block consisting of conv1 and res1 #
        ############################################################
        
        self.conv1 = [torch.nn.ReLU()]
        if upsample > 1:
            self.conv1 += [torch.nn.Upsample(scale_factor=upsample)]
        self.conv1 += [
            conv_type(input_dim, output_dim, kernel_size=kernel_size, padding=get_padding(kernel_size)),
            torch.nn.ReLU(),
            conv_type(output_dim, output_dim, kernel_size=kernel_size, dilation=3, padding=get_padding(kernel_size, 3))
            ]

        self.res1 = [torch.nn.Upsample(scale_factor=upsample)] if upsample > 1 else []
        self.res1 += [conv_type(input_dim, output_dim, kernel_size=1)]

        ####################################################
        # Create second residual block consisting of conv2 #
        ####################################################
        self.conv2 = [
            torch.nn.ReLU(),
            conv_type(
                output_dim,
                output_dim,
                kernel_size=kernel_size,
                dilation=9,
                padding=get_padding(kernel_size, 9)),
            torch.nn.ReLU(),
            conv_type(
                output_dim,
                output_dim,
                kernel_size=kernel_size,
                dilation=27,
                padding=get_padding(kernel_size, 27))]

        # Convert list of layers into torch.nn.Sequential
        self.conv1 = torch.nn.Sequential(*self.conv1)
        self.res1 = torch.nn.Sequential(*self.res1)
        self.conv2 = torch.nn.Sequential(*self.conv2)

    def forward(self, x): # x has shape (batchsize=1, dim, length)
        '''
        Args:
            x: (batchsize, dim, length)
        '''
        x = self.conv1(x) + self.res1(x)
        x = x + self.conv2(x)
        return x


class ResBlock(nn.Module):
    '''
    Gaddy and Klein, 2021, https://arxiv.org/pdf/2106.01933.pdf 
    Original code:
        https://github.com/dgaddy/silent_speech/blob/master/transformer.py
    '''
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = nn.BatchNorm1d(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        return F.relu(x + res)


class TransformerEncoderLayer(nn.Module):
    # Adapted from pytorch source
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
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, relative_positional=True, relative_positional_distance=100):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout, relative_positional=relative_positional, relative_positional_distance=relative_positional_distance)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model=256, n_head=4, dropout=0.1, relative_positional=True, relative_positional_distance=100):
    super().__init__()
    self.d_model = d_model
    self.n_head = n_head
    d_qkv = d_model // n_head
    assert d_qkv * n_head == d_model, 'd_model must be divisible by n_head'
    self.d_qkv = d_qkv

    self.w_q = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
    self.w_k = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
    self.w_v = nn.Parameter(torch.Tensor(n_head, d_model, d_qkv))
    self.w_o = nn.Parameter(torch.Tensor(n_head, d_qkv, d_model))
    nn.init.xavier_normal_(self.w_q)
    nn.init.xavier_normal_(self.w_k)
    nn.init.xavier_normal_(self.w_v)
    nn.init.xavier_normal_(self.w_o)

    self.dropout = nn.Dropout(dropout)

    if relative_positional:
        self.relative_positional = LearnedRelativePositionalEmbedding(relative_positional_distance, n_head, d_qkv, True)
    else:
        self.relative_positional = None

  def forward(self, x):
    """Runs the multi-head self-attention layer.
    Args:
      x: the input to the layer, a tensor of shape [length, batch_size, d_model]
    Returns:
      A single tensor containing the output from this layer
    """

    q = torch.einsum('tbf,hfa->bhta', x, self.w_q)
    k = torch.einsum('tbf,hfa->bhta', x, self.w_k)
    v = torch.einsum('tbf,hfa->bhta', x, self.w_v)
    logits = torch.einsum('bhqa,bhka->bhqk', q, k) / (self.d_qkv ** 0.5)

    if self.relative_positional is not None:
        q_pos = q.permute(2,0,1,3) #bhqd->qbhd
        l,b,h,d = q_pos.size()
        position_logits, _ = self.relative_positional(q_pos.reshape(l,b*h,d))
        # (bh)qk
        logits = logits + position_logits.view(b,h,l,l)

    probs = F.softmax(logits, dim=-1)
    probs = self.dropout(probs)
    o = torch.einsum('bhqk,bhka->bhqa', probs, v)
    out = torch.einsum('bhta,haf->tbf', o, self.w_o)
    return out


class LearnedRelativePositionalEmbedding(nn.Module):
    # from https://github.com/pytorch/fairseq/pull/2225/commits/a7fb63f2b84d5b20c8855e9c3372a95e5d0ea073
    """
    This module learns relative positional embeddings up to a fixed
    maximum size. These are masked for decoder and unmasked for encoder
    self attention.
    By default the embeddings are added to keys, but could be added to
    values as well.
    Args:
        max_relative_pos (int): the maximum relative positions to compute embeddings for
        num_heads (int): number of attention heads
        embedding_dim (int): depth of embeddings
        unmasked (bool): if the attention is unmasked (for transformer encoder)
        heads_share_embeddings (bool): if heads share the same relative positional embeddings
        add_to_values (bool): compute embeddings to be added to values as well
    """

    def __init__(
            self,
            max_relative_pos: int,
            num_heads: int,
            embedding_dim: int,
            unmasked: bool = False,
            heads_share_embeddings: bool = False,
            add_to_values: bool = False):
        super().__init__()
        self.max_relative_pos = max_relative_pos
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.unmasked = unmasked
        self.heads_share_embeddings = heads_share_embeddings
        self.add_to_values = add_to_values
        num_embeddings = (
            2 * max_relative_pos - 1
            if unmasked
            else max_relative_pos
        )
        embedding_size = (
            [num_embeddings, embedding_dim, 1]
            if heads_share_embeddings
            else [num_heads, num_embeddings, embedding_dim, 1]
        )
        if add_to_values:
            embedding_size[-1] = 2
        initial_stddev = embedding_dim**(-0.5)
        self.embeddings = nn.Parameter(torch.zeros(*embedding_size))
        nn.init.normal_(self.embeddings, mean=0.0, std=initial_stddev)

    def forward(self, query, saved_state=None):
        """
        Computes relative positional embeddings to be added to keys (and optionally values),
        multiplies the embeddings for keys with queries to create positional logits,
        returns the positional logits, along with embeddings for values (optionally)
        which could be added to values outside this module.
        Args:
            query (torch.Tensor): query tensor
            saved_state (dict): saved state from previous time step
        Shapes:
            query: `(length, batch_size*num_heads, embed_dim)`
        Returns:
            tuple(torch.Tensor):
                - positional logits
                - relative positional embeddings to be added to values
        """
        # During inference when previous states are cached
        if saved_state is not None and "prev_key" in saved_state:
            assert not self.unmasked, "This should only be for decoder attention"
            length = saved_state["prev_key"].shape[-2] + 1  # `length - 1` keys are cached,
                                                            # `+ 1` for the current time step
            decoder_step = True
        else:
            length = query.shape[0]
            decoder_step = False

        used_embeddings = self.get_embeddings_for_query(length)

        values_embeddings = (
            used_embeddings[..., 1]
            if self.add_to_values
            else None
        )
        positional_logits = self.calculate_positional_logits(query, used_embeddings[..., 0])
        positional_logits = self.relative_to_absolute_indexing(positional_logits, decoder_step)
        return (positional_logits, values_embeddings)

    def get_embeddings_for_query(self, length):
        """
        Extract the required embeddings. The maximum relative position between two time steps is
        `length` for masked case or `2*length - 1` for the unmasked case. If `length` is greater than
        `max_relative_pos`, we first pad the embeddings tensor with zero-embeddings, which represent
        embeddings when relative position is greater than `max_relative_pos`. In case `length` is
        less than `max_relative_pos`, we don't use the first `max_relative_pos - length embeddings`.
        Args:
            length (int): length of the query
        Returns:
            torch.Tensor: embeddings used by the query
        """
        pad_length = max(length - self.max_relative_pos, 0)
        start_pos = max(self.max_relative_pos - length, 0)
        if self.unmasked:
            with torch.no_grad():
                padded_embeddings = nn.functional.pad(
                    self.embeddings,
                    (0, 0, 0, 0, pad_length, pad_length)
                )
            used_embeddings = padded_embeddings.narrow(-3, start_pos, 2*length - 1)
        else:
            with torch.no_grad():
                padded_embeddings = nn.functional.pad(
                    self.embeddings,
                    (0, 0, 0, 0, pad_length, 0)
                )
            used_embeddings = padded_embeddings.narrow(-3, start_pos, length)
        return used_embeddings

    def calculate_positional_logits(self, query, relative_embeddings):
        """
        Multiplies query with the relative positional embeddings to create relative
        positional logits
        Args:
            query (torch.Tensor): Input tensor representing queries
            relative_embeddings (torch.Tensor): relative embeddings compatible with query
        Shapes:
            query: `(length, batch_size*num_heads, embed_dim)` if heads share embeddings
                   else `(length, batch_size, num_heads, embed_dim)`
            relative_embeddings: `(max_allowed_relative_positions, embed_dim)` if heads share embeddings
                                 else `(num_heads, max_allowed_relative_positions, embed_dim)`
                                 where `max_allowed_relative_positions` is `length` if masked
                                 else `2*length - 1`
        Returns:
            torch.Tensor: relative positional logits
        """
        if self.heads_share_embeddings:
            positional_logits = torch.einsum("lbd,md->lbm", query, relative_embeddings)
        else:
            query = query.view(query.shape[0], -1, self.num_heads, self.embedding_dim)
            positional_logits = torch.einsum("lbhd,hmd->lbhm", query, relative_embeddings)
            positional_logits = positional_logits.contiguous().view(
                positional_logits.shape[0], -1, positional_logits.shape[-1]
            )
        # mask out tokens out of range
        length = query.size(0)
        if length > self.max_relative_pos:
            # there is some padding
            pad_length = length - self.max_relative_pos
            positional_logits[:,:,:pad_length] -= 1e8
            if self.unmasked:
                positional_logits[:,:,-pad_length:] -= 1e8
        return positional_logits

    def relative_to_absolute_indexing(self, x, decoder_step):
        """
        Index tensor x (relative positional logits) in terms of absolute positions
        rather than relative positions. Last dimension of x represents relative position
        with respect to the first dimension, whereas returned tensor has both the first
        and last dimension indexed with absolute positions.
        Args:
            x (torch.Tensor): positional logits indexed by relative positions
            decoder_step (bool): is this is a single decoder step (during inference)
        Shapes:
            x: `(length, batch_size*num_heads, length)` for masked case or
               `(length, batch_size*num_heads, 2*length - 1)` for unmasked
        Returns:
            torch.Tensor: positional logits represented using absolute positions
        """
        length, bsz_heads, _ = x.shape

        if decoder_step:
            return x.contiguous().view(bsz_heads, 1, -1)

        if self.unmasked:
            x = nn.functional.pad(
                x,
                (0, 1)
            )
            x = x.transpose(0, 1)
            x = x.contiguous().view(bsz_heads, length * 2 * length)
            x = nn.functional.pad(
                x,
                (0, length - 1)
            )
            # Reshape and slice out the padded elements.
            x = x.view(bsz_heads, length + 1, 2*length - 1)
            return x[:, :length, length-1:]
        else:
            x = nn.functional.pad(
                x,
                (1, 0)
            )
            x = x.transpose(0, 1)
            x = x.contiguous().view(bsz_heads, length+1, length)
            return x[:, 1:, :]


class PastFCEncoder(torch.nn.Module):
    '''
    Autoregressive class in CARGAN
    https://github.com/descriptinc/cargan/blob/master/cargan/model/condition.py#L6
    '''
    def __init__(self, input_len=512, hidden_dim=256, output_dim=128):
        '''
        Args:
            input_len: the number of samples of autoregressive conditioning
        '''
        super().__init__()

        model = [
            torch.nn.Linear(input_len, hidden_dim),
            torch.nn.LeakyReLU(.1)]
        for _ in range(3):
            model.extend([
                torch.nn.Linear(
                    hidden_dim,
                    hidden_dim),
                torch.nn.LeakyReLU(.1)])
        model.append(
            torch.nn.Linear(hidden_dim, output_dim))
        self.model = torch.nn.Sequential(*model)
    
    def forward(self, x):
        '''
        Args:
            x: shape (batch_size, 1, input_len)

        Return:
            shape (batch_size, output_dim)
        '''
        x = x.reshape(x.shape[0], -1)
        return self.model(x)
        # return self.model(x.squeeze(1))


class PastSeqEncoder(torch.nn.Module):
    def __init__(self, output_dim=128, dropout=0.2, elayers=2, ffdim=512):
        super().__init__()
        self.l_conv1ds = nn.ModuleList([
            ResBlock(1, output_dim, 1),
        ])
        encoder_layer = TransformerEncoderLayer(d_model=output_dim, nhead=8, \
                            relative_positional=True, relative_positional_distance=100, \
                            dim_feedforward=ffdim, dropout=dropout)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, elayers)
    
    def forward(self, x):
        '''
        Args:
            x: shape (batch_size, 1, seq_len)

        Return:
            shape (batch_size, output_dim, seq_len)
        '''
        tmp = x
        for l_conv in self.l_conv1ds:
            tmp = l_conv(tmp) # (batchsize, dim, seq_len)
        tmp = tmp.permute(2, 0, 1)  # (seq_len, batchsize, dim)
        tmp = self.transformer(tmp) # (seq_len, batchsize, dim)
        tmp = tmp.permute(1, 2, 0)  # (batchsize, output_dim, seq_len)        
        return tmp