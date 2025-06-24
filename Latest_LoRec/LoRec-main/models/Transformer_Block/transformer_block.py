import torch.nn as nn

from models.Transformer_Block.multihead_attention import MultiHeadedAttention
from models.Transformer_Block.sublayer import SublayerConnection
from models.Transformer_Block.feed_forward import PositionwiseFeedForward


class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer   1024
        :param attn_heads: head sizes of multi-head attention   2
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size  128
        :param dropout: dropout rate    0.1
        """
        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask=None): #x.size()=torch.Size([567, 52, 1024])
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask)) #LayerNorm(x) → 自注意力 → Dropout → 残差连接
        x = self.output_sublayer(x, self.feed_forward) #LayerNorm(x) → 前馈网络 → Dropout → 残差连接
        return self.dropout(x)
