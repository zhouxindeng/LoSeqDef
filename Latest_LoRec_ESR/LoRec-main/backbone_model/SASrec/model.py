import pickle
import numpy as np
import torch
from torch import nn
from backbone_model.SASrec.encoders import User_Encoder, MLP_Layers
from torch.nn.init import xavier_normal_, constant_
from backbone_model.SASrec.GNN import UserGNN
import torch.nn.functional as F


class LLMSASrec(torch.nn.Module):
    def __init__(self, config):
        super(LLMSASrec, self).__init__()
        self.config = config
        self.max_seq_len = config["maxlen"] + 1

        self.fc = MLP_Layers(word_embedding_dim=config["LLM_size"],
                                 item_embedding_dim=config["hidden_units"],
                                 layers=[config["hidden_units"]] * (config["dnn_layer"] + 1),   #表示 MLP 有 1 层，每层 512 个节点。
                                 drop_rate=config["dropout_rate"])

        self.user_encoder = User_Encoder(
            item_num=config["n_items"],
            max_seq_len=config["maxlen"],
            item_dim=config["hidden_units"],
            num_attention_heads=config["num_heads"],
            dropout=config["dropout_rate"],
            n_layers=config["num_blocks"])


        self.gnn = UserGNN(
            input_dim=config["LLM_size"],
            hidden_dim=config["hidden_units"]*4,
            output_dim=config["hidden_units"],
            n_layers=2,
            dropout=config["dropout_rate"]
        )

    def _build_adjacency_matrix(self, interaction_mask, device):
        # Convert interaction sequences to user-item graph
        batch_size, seq_len = interaction_mask.shape
        
        # Create a similarity matrix based on co-occurrence
        adj = torch.zeros(batch_size, seq_len, seq_len).to(device)
        
        # For each user, create connections between items in their sequence
        for i in range(batch_size):
            # Get valid positions in the sequence (where mask is 1)
            valid_pos = (interaction_mask[i] == 1).nonzero(as_tuple=True)[0]
            if len(valid_pos) > 0:
                # Create connections between each pair of valid items
                for p1 in valid_pos:
                    for p2 in valid_pos:
                        adj[i, p1, p2] = 1.0
                
                # Normalize adjacency matrix for this user
                row_sum = adj[i].sum(dim=1, keepdim=True)
                row_sum[row_sum == 0] = 1  # Avoid division by zero
                adj[i] = adj[i] / row_sum
                
        return adj

    def forward(self, interaction_list, interaction_mask, neg_list):
        # log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])[:, :-1]
        # input_embs_all = self.fc(interaction_list)
        # input_logs_embs = input_embs_all[:, :-1, :]
        # target_pos_embs = input_embs_all[:, 1:, :]
        # target_neg_embs = self.fc(neg_list)[:, :-1, :]

        # prec_vec = self.user_encoder(input_logs_embs, log_mask, self.config["device"])

        # return prec_vec, target_pos_embs, target_neg_embs
        # pos_score = (prec_vec * target_pos_embs).sum(-1)
        # neg_score = (prec_vec * target_neg_embs).sum(-1)

        # return pos_score, neg_score
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.fc(interaction_list)   # LLM特征投影 [bathsize, max_len+1, LLM_size] -> [bathsize, max_len+1, layers(512)]
        input_logs_embs = input_embs_all
        target_pos_embs = input_embs_all[:, 1:, :]
        target_neg_embs = self.fc(neg_list)[:, :-1, :]  #[bathsize, max_len, LLM_size]

        adj_matrix=self._build_adjacency_matrix(log_mask,self.config["device"])
        gnn_vec = self.gnn(interaction_list, adj_matrix) ##[bathsize, max_len, LLM_size]
        similarity = F.cosine_similarity(input_logs_embs, gnn_vec, dim=2).sum(dim=1)/input_logs_embs.shape[1]
        similarity=similarity.mean()

        prec_vec = self.user_encoder(input_logs_embs, log_mask, self.config["device"])    # transformer序列编码结果（含LLM特征）torch.Size([512, 51, 512])

        return prec_vec[:, :-1, :], target_pos_embs, target_neg_embs, torch.cat((input_embs_all, prec_vec), dim=2), similarity  # 特征拼接：LLM原始特征 + 序列编码结果 [512, 51, 2*512]
    
    def predict(self, interaction_list, interaction_mask, item_indices):#interaction_list:torch.Size([512, 51, 5120]),interaction_mask:[512,51],item_indices:torch.Size([28638,5120])
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_logs_embs = self.fc(interaction_list)#torch.Size([512, 51, 512])
        item_embs = self.fc(item_indices)#torch.Size([28638, 512])

        prec_vec = self.user_encoder(input_logs_embs, log_mask, self.config["device"])[:, -1, :]#torch.Size([512, 51, 512])->torch.Size([512, 512])
        logits = item_embs.matmul(prec_vec.unsqueeze(-1)).squeeze(-1) #torch.Size([512, 28638])

        return logits
    
    def get_emb(self, interaction_list, interaction_mask):  #emb.size()=torch.Size([567, 51, 5120]),mask.size()=[567,51]
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.fc(interaction_list)  #torch.Size([567, 51, 512])
        input_logs_embs = input_embs_all

        prec_vec = self.user_encoder(input_logs_embs, log_mask, self.config["device"])

        pos_logits = (prec_vec[:,:-1,:] * input_logs_embs[:,1:,]).sum(-1) #逐元素相乘，并沿最后一个维度求和，pos_logits.size()=torch.Size([567, 51])
        pos_logits = torch.cat((pos_logits,pos_logits[:,-1].unsqueeze(1)),dim=1)

        #logits = (1-torch.sigmoid(pos_logits)).unsqueeze(2).expand(-1, -1, 2)
        logits = (1-torch.sigmoid(pos_logits)).unsqueeze(2)

        return torch.cat((input_embs_all, logits, prec_vec, logits), dim=2)

class BasicSASrec(torch.nn.Module):
    def __init__(self, config):
        super(BasicSASrec, self).__init__()
        self.config = config
        self.max_seq_len = config["maxlen"] + 1
        self.item_emb = torch.nn.Embedding(config["n_items"]+1, self.config["hidden_units"], padding_idx=0)
        xavier_normal_(self.item_emb.weight.data[1:])

        self.user_encoder = User_Encoder(
            item_num=config["n_items"],
            max_seq_len=config["maxlen"],
            item_dim=config["hidden_units"],
            num_attention_heads=config["num_heads"],
            dropout=config["dropout_rate"],
            n_layers=config["num_blocks"])


    def forward(self, interaction_list, interaction_mask, neg_list):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.item_emb(torch.LongTensor(np.array(interaction_list)).to(self.config["device"]))
        input_logs_embs = input_embs_all
        target_pos_embs = input_embs_all[:, 1:, :]
        target_neg_embs = self.item_emb(torch.LongTensor(np.array(neg_list)).to(self.config["device"]))[:, :-1, :]

        prec_vec = self.user_encoder(input_logs_embs, log_mask, self.config["device"])

        return prec_vec[:, :-1, :], target_pos_embs, target_neg_embs, torch.cat((input_embs_all, prec_vec), dim=2)
        # return prec_vec, target_pos_embs, target_neg_embs, prec_vec
        # pos_score = (prec_vec * target_pos_embs).sum(-1)
        # neg_score = (prec_vec * target_neg_embs).sum(-1)

        # return pos_score, neg_score
    
    def predict(self, interaction_list, interaction_mask, item_indices):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_logs_embs = self.item_emb(torch.LongTensor(np.array(interaction_list)).to(self.config["device"]))
        item_embs = self.item_emb(torch.LongTensor(np.array(item_indices)).to(self.config["device"]))

        prec_vec = self.user_encoder(input_logs_embs, log_mask, self.config["device"])[:, -1, :]
        logits = item_embs.matmul(prec_vec.unsqueeze(-1)).squeeze(-1)

        return logits

    def get_emb(self, interaction_list, interaction_mask):
        log_mask = torch.FloatTensor(interaction_mask).to(self.config["device"])
        input_embs_all = self.item_emb(torch.LongTensor(np.array(interaction_list)).to(self.config["device"]))
        input_logs_embs = input_embs_all

        prec_vec = self.user_encoder(input_logs_embs, log_mask, self.config["device"])

        return torch.cat((input_embs_all, prec_vec), dim=2)