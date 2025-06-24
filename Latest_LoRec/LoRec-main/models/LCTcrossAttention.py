import torch
import torch.nn as nn
from models.Transformer_Block.position_emb import PositionEmb
from models.Transformer_Block.transformer_block import TransformerBlock
import torch.nn.functional as F


class MLP_Layers(nn.Module):
    def __init__(self, word_embedding_dim, item_embedding_dim, layers, drop_rate):
        super(MLP_Layers, self).__init__()
        self.layers = layers
        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(zip(self.layers[:-1], self.layers[1:])):#[1024,1024]
            mlp_modules.append(nn.Dropout(p=drop_rate))
            mlp_modules.append(nn.Linear(input_size, output_size))
            mlp_modules.append(nn.GELU())
        self.mlp_layers = nn.Sequential(*mlp_modules)
        self.fc = nn.Linear(word_embedding_dim, item_embedding_dim)
        self.activate = nn.GELU()   
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    def forward(self, sample_items):
        sample_items = self.activate(self.fc(sample_items)) #Linner:[5120,1024]->[1024,1024]
        return self.mlp_layers(sample_items)
    

class ProjLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, drop_rate) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, 10*hidden_dim)
        self.linear2= nn.Linear(10*hidden_dim,4*hidden_dim)
        self.linear3= nn.Linear(4*hidden_dim,hidden_dim)
        self.proj = nn.Linear(hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(10*hidden_dim)
        self.bn2 = nn.BatchNorm1d(4*hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.bn4 = nn.BatchNorm1d(1)
        self.drop = nn.Dropout(p=drop_rate)
        self.active = nn.GELU()
    
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            # if module.bias is not None:
            #     nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        x = self.drop(self.active(self.bn1(self.linear(x))))  #linear:[1024,128]->bn:[128]->drop(0.1)->gelu
        x = self.drop(self.active(self.bn2(self.linear2(x)))) 
        x = self.drop(self.active(self.bn3(self.linear3(x))))  
        return self.bn4(self.proj(x)) #linear:[128,1]->bn[1] 

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=8, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        # 定义query, key, value的线性投影层
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight.data)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)
    
    def forward(self, x, context):
        batch_size = x.size(0)
        
        # 线性投影 [567,1,1024]->[567,-1,2,512]->[567,2,1,512]
        q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(context).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        context_vector = torch.matmul(attn_weights, v)
        context_vector = context_vector.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # 输出投影
        output = self.proj(context_vector)
        
        return output.squeeze(1)  # 返回形状为 [batch_size, hidden_dim]

class LCTer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.fc = MLP_Layers(word_embedding_dim=config["LLM_size"],
                                item_embedding_dim=2 * config["hidden_units"],
                                layers=[2 * config["hidden_units"]] * (config["dnn_layer"] + 1), #[1024]
                                drop_rate=config["dropout_rate"])

        self.transformer_layers = nn.ModuleList([TransformerBlock(2 * config["hidden_units"], config["heads"], config["ffd_hidden"], config["dropout_rate"])
                                          for _ in range(config.get("block_num", 1))])
        
        self.seq_emb = torch.nn.Embedding(1, 2 * self.config["hidden_units"])#embedding[1,1024]
        nn.init.xavier_normal_(self.seq_emb.weight.data)

        self.proj = ProjLayer(in_dim=2 * config["hidden_units"], hidden_dim=128, drop_rate=config["dropout_rate"])

        self.sigmoid = nn.Sigmoid()
        
        self.position_ecd = PositionEmb(2 * config["hidden_units"], config.get("max_len", 50))

        self.cross_attention = CrossAttention(
            hidden_dim=2 * config["hidden_units"], 
            num_heads=config["heads"], 
            dropout_rate=config["dropout_rate"]
        )

    def forward(self, seq, detections=None, mask=None):
        detections = self.fc(detections)
        seq = seq + self.position_ecd(seq)  #[567, 51, 512]+[1, 51, 512]
        seq_embs = self.seq_emb(torch.tensor([0]).to(seq.device)) #torch.Size([1, 1024])
        seq_embs = seq_embs.unsqueeze(1).expand(seq.size(0), -1, -1)#torch.Size([567, 1, 1024])
        x = torch.cat((seq_embs, seq), dim=1)   #将特殊标记拼接到输入序列开头。torch.Size([567, 52, 1024])
    

        if mask is not None:
            ones_tensor = torch.ones(seq_embs.size(0), 1, seq_embs.size(2), dtype=torch.bool).to(seq.device)#torch.Size([567, 1, 1024])
            mask = torch.FloatTensor(mask).unsqueeze(-1).expand_as(seq).to(seq.device)#torch.Size([567, 51, 1024])
            mask = torch.cat((ones_tensor, mask), dim=1)#torch.Size([567, 52, 1024])
        

        for layer in self.transformer_layers:
            if mask is not None:
                x *= mask
            x = layer(x)   #x.size()=torch.Size([567, 52, 1024])
        
        x = x[:,0,:].squeeze() #torch.Size([567, 1024])
        
        #x与detecions实现交叉注意力
        cross_output = self.cross_attention(x.unsqueeze(1), detections.unsqueeze(1))
        x = x + cross_output
        
        score = self.proj(x) #torch.Size([567, 1])

        return self.sigmoid(score)
