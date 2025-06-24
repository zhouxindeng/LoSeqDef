import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_, constant_
from backbone_model.SASrec.modules import TransformerEncoder


class MLP_Layers(torch.nn.Module):
    def __init__(self, word_embedding_dim, item_embedding_dim, layers, drop_rate):
        super(MLP_Layers, self).__init__()
        self.layers = layers
        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):
            mlp_modules.append(nn.Dropout(p=drop_rate))
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.GELU())
        """ self.layers 是一个包含每一层神经元数目的列表，通常是 [input_size, hidden_size_1, hidden_size_2, ..., output_size]。
        self.layers[:-1] 表示除了最后一层外的所有层，这样可以获得每层的输入大小（input_size）。
        self.layers[1:] 表示从第二层到最后一层，得到每层的输出大小（output_size）。
        zip 会将这两个列表“配对”，即把每一层的输入和输出大小组成一对元组。 """
        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        self.activate = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):    
            xavier_normal_(module.weight.data)  #使用 xavier_normal_（Xavier 正态分布）初始化。
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)  #对偏置项使用常数 0 初始化

    def forward(self, sample_items):
        sample_items = self.activate(self.fc(sample_items))
        return self.mlp_layers(sample_items)


class User_Encoder(torch.nn.Module):
    def __init__(self, item_num, max_seq_len, item_dim, num_attention_heads, dropout, n_layers):
        super(User_Encoder, self).__init__()
        self.transformer_encoder = TransformerEncoder(n_vocab=item_num, n_position=max_seq_len,
                                                      d_model=item_dim, n_heads=num_attention_heads,
                                                      dropout=dropout, n_layers=n_layers)   #使用Transformer模型的编码部分，head=2
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, input_embs, log_mask, local_rank): #input_embs.size()=torch.Size([512, 50, 512])
        att_mask = (log_mask != 0)
        att_mask = att_mask.unsqueeze(1).unsqueeze(2)  # torch.bool,Size([512, 1, 1, 51])
        att_mask = torch.tril(att_mask.expand((-1, -1, log_mask.size(-1), -1))).to(local_rank) #下三角Size([512, 1, 51, 51])
        att_mask = torch.where(att_mask, 0., -1e9)  #Transformer掩码
        return self.transformer_encoder(input_embs, log_mask, att_mask) 