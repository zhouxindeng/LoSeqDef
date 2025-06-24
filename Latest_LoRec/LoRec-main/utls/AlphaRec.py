import json
import os
import time
import numpy as np
import torch
import torch.optim as optim
import random
from torch.utils.data import DataLoader
from datetime import datetime
import torch.nn.functional as F
from torch import nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import CheckButtons
from utls.mydataset import AlphaDataset
from utls.utilize import custom_loss, slice_lists


def infoNCE(similarity_scores, pos_indices, neg_indices, temperature=0.1):
    """
    InfoNCE损失函数
    Args:
        similarity_scores: [batch_size, n_items] 用户与所有items的相似度分数
        pos_indices: [batch_size] 正样本的索引
        neg_indices: [batch_size, neg_samples] 负样本的索引  
        temperature: 温度参数
    """
    if not isinstance(pos_indices, torch.Tensor):
            pos_indices = torch.tensor(pos_indices, device=similarity_scores.device)
    if not isinstance(neg_indices, torch.Tensor):
            neg_indices = torch.tensor(neg_indices, device=similarity_scores.device)
    # 获取正样本的相似度分数
    pos_scores = similarity_scores.gather(1, pos_indices.unsqueeze(1)).squeeze(1)  # [batch_size]
    
    # 获取负样本的相似度分数
    neg_scores = similarity_scores.gather(1, neg_indices)  # [batch_size, neg_samples]
    
    # 应用温度缩放
    pos_scores = pos_scores / temperature
    neg_scores = neg_scores / temperature
    
    # 计算InfoNCE损失
    numerator = torch.exp(pos_scores)
    denominator = numerator + torch.sum(torch.exp(neg_scores), dim=1)
    
    loss = -torch.log(numerator / denominator)
    return torch.mean(loss)

class AlphaRec(nn.Module):
    def __init__(self, config):
        super(AlphaRec, self).__init__()
        self.config = config
        self.linear = nn.Linear(config["LLM_size"], 2*config["hidden_units"])
        self.item_features = None
    
    def forward(self, all_idx, all_interaction_list, mask):
        # 确保输入是tensor且在正确设备上
        self.item_features=self.linear(all_idx)
        if not isinstance(all_interaction_list, torch.Tensor):
            all_interaction_list = torch.tensor(all_interaction_list, dtype=torch.long,device=self.item_features.device)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32,device=self.item_features.device)
        
        batch_size, max_len = all_interaction_list.size()
        hidden_units = self.item_features.size(1)
        
        # 向量化处理：展平所有交互序列
        flat_interactions = all_interaction_list.view(-1)  # [batch_size * max_len]
        
        # 获取所有交互的特征
        all_interaction_features = self.item_features[flat_interactions]  # [batch_size * max_len, hidden_units]
        
        # 重新reshape并应用mask
        interaction_features = all_interaction_features.view(batch_size, max_len, hidden_units)  # [batch_size, max_len, hidden_units]
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, hidden_units)  # [batch_size, max_len, hidden_units]
        
        # 应用mask：将无效位置设为0
        masked_features = interaction_features * mask_expanded  # [batch_size, max_len, hidden_units]
        
        # 计算每个用户的特征和（避免除零）
        feature_sum = torch.sum(masked_features, dim=1)  # [batch_size, hidden_units]
        valid_counts = torch.sum(mask, dim=1, keepdim=True)  # [batch_size, 1]
        valid_counts = torch.clamp(valid_counts, min=1.0)  # 避免除零
        
        # 计算用户表示（平均）
        user_representations = feature_sum / valid_counts  # [batch_size, hidden_units]
        # 计算用户表示与所有items的余弦相似度
        # 归一化
        user_representations = F.normalize(user_representations, dim=-1)
        item_features_norm = F.normalize(self.item_features, dim=-1)
        
        # 计算相似度
        similarity = torch.matmul(user_representations, item_features_norm.T)  # [batch_size, n_items+1]
        
        return similarity
    
class VisualizationModule:
    """处理t-SNE降维和可视化的模块"""
    
    def __init__(self, config):
        self.config = config
        self.semantic_tsne = None  # 语义空间的t-SNE结果
        self.representation_tsne = None  # 表示空间的t-SNE结果
        
    def perform_tsne(self, semantic_features, representation_features, perplexity=30, n_iter=1000, random_state=42):
        """
        对语义空间和表示空间进行t-SNE降维
        
        Args:
            semantic_features: [n_items+1, LLM_size] 原始LLM特征
            representation_features: [n_items+1, 2*hidden_units] 线性映射后的特征
            perplexity: t-SNE的困惑度参数
            n_iter: 迭代次数
            random_state: 随机种子
        """
        print("开始t-SNE降维...")
        
        # 转换为numpy数组
        if torch.is_tensor(semantic_features):
            semantic_features = semantic_features.detach().cpu().numpy()
        if torch.is_tensor(representation_features):
            representation_features = representation_features.detach().cpu().numpy()
        
        # 对语义空间进行t-SNE
        print("对语义空间进行t-SNE降维...")
        tsne_semantic = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                           random_state=random_state, verbose=1)
        self.semantic_tsne = tsne_semantic.fit_transform(semantic_features)
        
        # 对表示空间进行t-SNE
        print("对表示空间进行t-SNE降维...")
        tsne_representation = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                                 random_state=random_state, verbose=1)
        self.representation_tsne = tsne_representation.fit_transform(representation_features)
        
        print("t-SNE降维完成!")
        
    def visualize_spaces(self, connection_indices=None, item_names=None, figsize=(15, 6), save_path=None):
        """
        可视化语义空间和表示空间，并可选择性地显示连线
        
        Args:
            connection_indices: 要显示连线的项目索引列表，None表示不显示连线
            item_names: 项目名称列表，用于标注
            figsize: 图像大小
            save_path: 保存路径，None表示不保存
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif']=['Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        if self.semantic_tsne is None or self.representation_tsne is None:
            raise ValueError("请先调用perform_tsne()进行降维!")
        
        # 设置图像风格
        plt.style.use('default')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.patch.set_facecolor('white')
        
        n_points = len(self.semantic_tsne)
        
        # 定义颜色
        gray_color = '#B3B3B3'  # RGB(179,179,179)
        
        # 如果没有指定连线索引，所有点都是灰色
        if connection_indices is None or len(connection_indices) == 0:
            # 语义空间可视化 - 全部灰色
            scatter1 = ax1.scatter(self.semantic_tsne[:, 0], self.semantic_tsne[:, 1], 
                                c=gray_color, alpha=0.5, s=5, edgecolors='none', linewidth=0.5)
            
            # 表示空间可视化 - 全部灰色
            scatter2 = ax2.scatter(self.representation_tsne[:, 0], self.representation_tsne[:, 1], 
                                c=gray_color, alpha=0.5, s=5, edgecolors='none', linewidth=0.5)
        else:
            # 为连线的点生成不同颜色
            highlight_colors = plt.cm.Set1(np.linspace(0, 1, len(connection_indices)))
            
            # 语义空间可视化
            # 先绘制所有灰色点
            scatter1 = ax1.scatter(self.semantic_tsne[:, 0], self.semantic_tsne[:, 1], 
                                c=gray_color, alpha=0.5, s=5, edgecolors='none', linewidth=0.5)
            
            # 再绘制高亮点
            for i, idx in enumerate(connection_indices):
                if idx < len(self.semantic_tsne):
                    ax1.scatter(self.semantic_tsne[idx, 0], self.semantic_tsne[idx, 1], 
                            c=[highlight_colors[i]], alpha=0.9, s=20, edgecolors='none', linewidth=1.5)
            
            # 表示空间可视化
            # 先绘制所有灰色点
            scatter2 = ax2.scatter(self.representation_tsne[:, 0], self.representation_tsne[:, 1], 
                                c=gray_color, alpha=0.5, s=5, edgecolors='none', linewidth=0.5)
            
            # 再绘制高亮点
            for i, idx in enumerate(connection_indices):
                if idx < len(self.representation_tsne):
                    ax2.scatter(self.representation_tsne[idx, 0], self.representation_tsne[idx, 1], 
                            c=[highlight_colors[i]], alpha=0.9, s=20, edgecolors='none', linewidth=1.5)
        
        # 设置标题和标签
        ax1.set_title('Language Space', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('t-SNE dim 1', fontsize=12)
        ax1.set_ylabel('t-SNE dim 2', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_facecolor('#f8f9fa')
        
        ax2.set_title('Behavior Space', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlabel('t-SNE dim 1', fontsize=12)
        ax2.set_ylabel('t-SNE dim 2', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_facecolor('#f8f9fa')
        
        # 添加连线（如果指定了索引）
        #if connection_indices is not None and len(connection_indices) > 0:
        #    self._add_connection_lines(fig, ax1, ax2, connection_indices)
        
        # 添加索引标注（如果指定了索引）
        if connection_indices is not None and len(connection_indices) > 0:
            self._add_index_labels(ax1, ax2, connection_indices)
        
        # 添加文本标注（如果提供了项目名称）
        if item_names is not None and connection_indices is not None:
            self._add_item_labels(ax1, ax2, connection_indices, item_names)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"图像已保存到: {save_path}")
        
        plt.show()
    
    # 添加新的索引标注方法
    def _add_index_labels(self, ax1, ax2, connection_indices):
        """为选中的项目添加索引标注"""
        for idx in connection_indices:
            if idx < len(self.semantic_tsne) and idx < len(self.representation_tsne):
                # 在语义空间添加索引标注
                ax1.annotate(str(idx), 
                        (self.semantic_tsne[idx, 0], self.semantic_tsne[idx, 1]),
                        xytext=(8, 8), textcoords='offset points', 
                        fontsize=9, alpha=0.9, color='black', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor='gray'))
                
                # 在表示空间添加索引标注
                ax2.annotate(str(idx), 
                        (self.representation_tsne[idx, 0], self.representation_tsne[idx, 1]),
                        xytext=(8, 8), textcoords='offset points', 
                        fontsize=9, alpha=0.9, color='black', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor='gray'))
        
    def _add_connection_lines(self, fig, ax1, ax2, connection_indices):
        """在两个子图之间添加连线"""
        # 获取两个子图的位置
        pos1 = ax1.get_position()
        pos2 = ax2.get_position()
        
        # 为连线生成颜色
        highlight_colors = plt.cm.Set1(np.linspace(0, 1, len(connection_indices)))
        
        # 为选中的项目添加连线
        for i, idx in enumerate(connection_indices):
            if idx < len(self.semantic_tsne) and idx < len(self.representation_tsne):
                # 计算连线的起点和终点（在figure坐标系中）
                x1_norm = pos1.x1  # 语义空间右边界
                x2_norm = pos2.x0  # 表示空间左边界
                
                # 将数据坐标转换为figure坐标
                y1_data = self.semantic_tsne[idx, 1]
                y2_data = self.representation_tsne[idx, 1]
                
                # 获取y轴的数据范围并转换为figure坐标
                y1_range = ax1.get_ylim()
                y2_range = ax2.get_ylim()
                
                y1_norm = pos1.y0 + (y1_data - y1_range[0]) / (y1_range[1] - y1_range[0]) * pos1.height
                y2_norm = pos2.y0 + (y2_data - y2_range[0]) / (y2_range[1] - y2_range[0]) * pos2.height
                
                # 添加连线，使用与点相同的颜色
                color = matplotlib.colors.to_hex(highlight_colors[i])
                line = plt.Line2D([x1_norm, x2_norm], [y1_norm, y2_norm], 
                                transform=fig.transFigure, color=color, alpha=0.8, linewidth=2)
                fig.lines.append(line)

    def _add_item_labels(self, ax1, ax2, connection_indices, item_names):
        """为选中的项目添加文本标注"""
        for idx in connection_indices:
            if idx < len(item_names) and idx < len(self.semantic_tsne):
                # 在语义空间添加标注
                ax1.annotate(item_names[idx], 
                        (self.semantic_tsne[idx, 0], self.semantic_tsne[idx, 1]),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, alpha=0.9, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='gray'))
                
                # 在表示空间添加标注
                ax2.annotate(item_names[idx], 
                        (self.representation_tsne[idx, 0], self.representation_tsne[idx, 1]),
                        xytext=(5, 5), textcoords='offset points', 
                        fontsize=10, alpha=0.9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='gray'))
        
    
    def save_tsne_results(self, save_path):
        """保存t-SNE降维结果"""
        if self.semantic_tsne is None or self.representation_tsne is None:
            raise ValueError("请先调用perform_tsne()进行降维!")
        
        results = {
            'semantic_tsne': self.semantic_tsne.tolist(),
            'representation_tsne': self.representation_tsne.tolist()
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"t-SNE结果已保存到: {save_path}")
    
    def load_tsne_results(self, load_path):
        """加载t-SNE降维结果"""
        with open(load_path, 'r') as f:
            results = json.load(f)
        
        self.semantic_tsne = np.array(results['semantic_tsne'])
        self.representation_tsne = np.array(results['representation_tsne'])
        
        print(f"t-SNE结果已从 {load_path} 加载")

class BasicTrainer:
    def __init__(self, trainer_config) -> None:
        self.config = trainer_config
        self.device = trainer_config['device']
        self.n_epochs = trainer_config['n_epochs']
        self.min_epochs = trainer_config['min_epochs']
        self.max_patience = trainer_config.get('patience', 50)
        self.val_interval = trainer_config.get('val_interval', 1)
    
    def _create_dataset(self, path):
        raise NotImplementedError
    
    def _create_dataloader(self):
        self.dataloader = DataLoader(self.dataset, batch_size=self.config["batch_size"], shuffle=True)

    def _create_model(self):
        raise NotImplementedError
    
    def _create_opt(self):
        raise NotImplementedError

    def _train_epoch(self, epoch):
        raise NotImplementedError
    
    def _eval_model(xself, epoch):
        raise NotImplementedError
    
    def test_model(self):
        raise NotImplementedError

    def _save_model(self, best_model_path):
        raise NotImplementedError
    
    def _load_model(self, best_model_path):
        raise NotImplementedError

    def _update_weight(self):
        raise NotImplementedError

    def _init_path(self, path=None):
        best_model_path = f"{self.config['checkpoints']}/AlphaRec/{self.config['dataset']}"
        if self.config["main_file"] != "":
            best_model_path = os.path.join(best_model_path, self.config["main_file"])
        if path is not None:
            best_model_path = path
        if not os.path.exists(best_model_path):
            os.makedirs(best_model_path, exist_ok=True)
        best_model_path = os.path.join(best_model_path, f"{'attack_'+ str(self.config['inject_persent']) if self.config['inject_user'] else 'normal'}_{datetime.now().strftime('%Y%m%d%H%M')}.pth")

        return best_model_path

    def train(self, path=None):
        patience = self.config["patience"]
        best_metrics = -1

        best_model_path = self._init_path(path=path)
        
        self.update_flag = False
        for epoch in range(self.n_epochs):
            self._train_epoch(epoch)

            if (epoch + 1) % self.config["val_interval"] == 0:
                avg_metrics = self._eval_model(epoch)
            
                if (epoch + 1) >= self.config["min_epochs"]:
                    if avg_metrics > best_metrics:
                        best_metrics = avg_metrics
                        # Save the best model
                        self._save_model(best_model_path)
                        patience = self.config["patience"]
                    else:
                        patience -= self.config["val_interval"]
                        if patience <= 0:
                            print('Early stopping!')
                            break
                
        self._load_model(best_model_path)
        # Test
        hr, ndcg = self.test_model()

        return hr, ndcg

class AlphaRecTrainer(BasicTrainer):
    def __init__(self, trainer_config) -> None:
        super().__init__(trainer_config)

        #self._create_dataset(f"/root/tf-logs/lc_LoRec/LoRec-main/data/{trainer_config['dataset']}")
        self._create_dataset(f"/Users/changliu/Documents/GraduationProject//Latest_LoRec/LoRec-main/data/{trainer_config['dataset']}")
        self._create_dataloader()
        self._create_model()
        self._create_opt()

        # 初始化可视化模块
        self.visualizer = VisualizationModule(trainer_config)

    def _create_dataset(self, path):
        self.dataset = AlphaDataset(path, self.config["LLM"], self.config, has_fake_user=self.config["with_lct"], max_len=self.config["max_interaction"])
    
    def _create_dataloader(self):
        return super()._create_dataloader()

    def _create_model(self):
        self.config["model_config"]["LLM_size"] = self.config["LLM_size"]
        self.config["model_config"]["n_users"] = self.dataset.n_users
        self.config["model_config"]["n_items"] = self.dataset.n_items
        self.model = AlphaRec(self.config["model_config"]).to(self.device)

        if torch.cuda.is_available() and self.config["use_gpu"]:
            self.model.cuda()
    
    def _create_opt(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    def extract_features_for_visualization(self):
        """
        提取用于可视化的特征
        返回语义空间特征和表示空间特征
        """
        self.model.eval()
        
        # 获取所有items的LLM embeddings
        all_idx = list(range(self.dataset.n_items + 1))
        semantic_features = self.dataset.get_labels_emb(all_idx)  # [n_items+1, LLM_size]
        
        # 通过模型的线性层获得表示空间特征
        with torch.no_grad():
            representation_features = self.model.linear(semantic_features)  # [n_items+1, 2*hidden_units]
        
        return semantic_features, representation_features
    
    def visualize_feature_spaces(self, connection_indices=None, item_names=None, save_visualization=None, save_tsne=None,load_tsne=None):
        """
        可视化语义空间和表示空间
        
        Args:
            connection_indices: 要显示连线的项目索引列表，None表示不显示连线
            item_names: 项目名称列表
            perform_tsne: 是否重新进行t-SNE降维
            save_visualization: 可视化图像保存路径
            save_tsne: t-SNE结果保存路径
        """
        print("开始特征空间可视化...")
        
        # 提取特征
        semantic_features, representation_features = self.extract_features_for_visualization()
        
        # 进行t-SNE降维（如果需要）
        if load_tsne:
            self.visualizer.load_tsne_results(load_tsne)
        else:
            self.visualizer.perform_tsne(semantic_features, representation_features)
            
            # 保存t-SNE结果（如果指定路径）
            if save_tsne:
                self.visualizer.save_tsne_results(save_tsne)
        
        # 可视化
        self.visualizer.visualize_spaces(connection_indices=connection_indices, 
                                       item_names=item_names, 
                                       save_path=save_visualization)
              
    def _train_epoch(self, epoch):
        start_t = time.time()
        epoch_loss = 0
        
        # 获取所有items的LLM embeddings并进行线性变换
        all_idx = list(range(self.dataset.n_items + 1))
        all_idx = self.dataset.get_labels_emb(all_idx)  # [n_items+1, LLM_size]

        for batch_data in self.dataloader:
            self.opt.zero_grad()
            self.model.train()
            
            # 获取训练批次数据
            seq, mask, neg, pos = self.dataset.get_train_batch(batch_data)
            # seq: [batch_size, max_len] 用户交互序列
            # mask: [batch_size, max_len] 掩码
            # neg: [batch_size, max_len] 负样本
            # pos: [batch_size] 正样本
            
            # 计算用户表示与所有items的相似度
            similarity = self.model(all_idx, seq, mask)  # [batch_size, n_items+1]
      
            # 计算InfoNCE损失
            loss = infoNCE(similarity, pos, neg, temperature=self.config.get("temperature", 0.1))
            
            loss.backward()
            self.opt.step()
            epoch_loss += loss.item()
            
        end_t = time.time()
        loss_text = f"Epoch {epoch}: Rec Loss: {epoch_loss:.4f}"
        print(loss_text + f", Time: {end_t-start_t:.2f}")

    
    def _eval_model(self, epoch):
        self.model.eval()
        hr_all = 0
        ndcg_all = 0
        
        for batch_data in self.dataloader:
            org_seq, mask, pos = self.dataset.get_val_batch(batch_data)
            hr, ndcg = self._eval_rec(org_seq, mask, pos)
            hr_all += hr
            ndcg_all += ndcg
        
        avg_hr, avg_ndcg = hr_all / self.dataset.n_users, ndcg_all / self.dataset.n_users
        print(f"Validation at epoch {epoch} - Hit Ratio@{self.config['top_k']}: {avg_hr:4f},  NDCG@{self.config['top_k']}: {avg_ndcg:4f}")

        return avg_hr
        
    def test_model(self, model_path=None):
        if model_path is not None:
            self._load_model(model_path)
        self.model.eval()
        hr_all = 0
        ndcg_all = 0
        cnt = 0
        
        for batch_data in self.dataloader:
            org_seq, mask, pos = self.dataset.get_test_batch(batch_data)
            if len(pos) == 0:
                continue
            cnt += len(pos)
            hr, ndcg = self._eval_rec(org_seq, mask, pos)
            hr_all += hr
            ndcg_all += ndcg
        
        avg_hr, avg_ndcg = hr_all / cnt, ndcg_all / cnt
        print(f"Test - Hit Ratio@{self.config['top_k']}: {avg_hr:4f},  NDCG@{self.config['top_k']}: {avg_ndcg:4f}")

        return avg_hr, avg_ndcg
    
    def _eval_rec(self, org_seq, mask, pos):
        self.model.eval()
        
        # 获取所有items的特征
        all_idx = list(range(self.dataset.n_items + 1))
        all_idx = self.dataset.get_labels_emb(all_idx)
        with torch.no_grad():
            all_logits = self.model(all_idx, org_seq, mask)
        #mask=torch.FloatTensor(mask).to(self.config["device"])
        #org_seq=torch.FloatTensor(org_seq).to(self.config["device"])
        # 排除用户已经交互过的items
        for i in range(len(org_seq)):
            user_interactions = org_seq[i]  # [max_len+1]
            user_mask = mask[i]  # [max_len+1]
            interactions_np = np.array(user_interactions)
            mask_np = np.array(user_mask)
            # 找到有效的交互位置
            valid_positions = np.where(mask_np > 0)[0]  # 使用numpy的where
        
            # 获取有效的item IDs
            valid_item_ids = interactions_np[valid_positions]

        
            # 将这些交互过的items的相似度设为负无穷
            for item_id in valid_item_ids:
                all_logits[i, int(item_id)] = float('-inf')
        
        # 排除padding item (索引0)
        all_logits = all_logits[:, 1:]
        
        HR = 0
        NDCG = 0
        _, sorted_indices = all_logits.sort(dim=1, descending=True)
        
        for user_idx, item_idx in enumerate(pos):
            rank = (sorted_indices[user_idx] == item_idx - 1).nonzero().item() + 1
            if rank <= self.config["top_k"]:
                HR += 1
                NDCG += 1 / np.log2(rank + 1)
        
        return HR, NDCG
    
    def _save_model(self, best_model_path):
        torch.save({
            'AlphaRec': self.model.state_dict(),
        }, best_model_path)
    
    def _load_model(self, best_model_path):
        checkpoint = torch.load(best_model_path)
        self.model.load_state_dict(checkpoint['AlphaRec'])