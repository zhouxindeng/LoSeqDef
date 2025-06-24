import json
import os
import random
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from utls.LLM4process import *

from tqdm import tqdm

import os
from openai import OpenAI

""" client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),  # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
) """


class BasicDataset(Dataset):
    def __init__(self, path, LLM, config, has_fake_user=False, max_len=1000) -> None:
        super().__init__()
        self.path = path
        self.LLM = LLM
        self.config = config
        self.has_fake_user = has_fake_user
        self.max_len = max_len

    def __len__(self):
        return self.n_users
    
    def __getitem__(self, index):
        return index

    def _item_fatures_LLM(self, item_features):
        item_pool = {}
        glo = globals()
        LLM_model = glo[f"{self.LLM}"](self.config)
        for idx, item in tqdm(item_features.items()):
            item_emb = LLM_model.general_LLM(item)          #LLM(promptu) item_emb.size=[n,LLM_size],item_pool[i].size=[LLM_size]
            item_pool[idx] = torch.mean(item_emb, dim=0)
        return item_pool
    
    def _detection_LLM(self, text):
        glo = globals()
        LLM_model = glo[f"{self.LLM}"](self.config)
        detection = LLM_model.general_LLM(text)
        return torch.mean(detection, dim=0)
    
    def _load_detection_fake_users(self, item_features):
        glo = globals()
        LLM_model = glo[f"{self.LLM}"](self.config)
        rs_scenario = f"In {self.config['recommendation_scenario']}, a user's interaction sequence is as follows:"
        question = ". Please assess the likelihood of this user being a fraudster.Just answer with 'High risk' or 'Low risk'."
        
        fake_user_detections = {}
        for user, interactions in tqdm(self.fake_data.items()):
            detection_text = rs_scenario
            for item in interactions[:-2]:
                detection_text += f" (Item title: {item_features[item]});"
            detection_text = detection_text[:-1] + question
            completion = client.chat.completions.create(
                model="deepseek-v3",  
                messages=[
                {'role': 'user', 'content': detection_text}
                ]
            )
            answer=completion.choices[0].message.content
            detection_emb = LLM_model.general_LLM(answer)
            fake_user_detections[user] = torch.mean(detection_emb, dim=0)
        
        return fake_user_detections

    def _load_detection_users(self, item_features):#目前的user_interaction中的user(0到users-1):item_id(1到n):v
        user_detections = {}
        glo = globals()
        LLM_model = glo[f"{self.LLM}"](self.config)
        rs_scenario = f"In {self.config['recommendation_scenario']}, a user's interaction sequence is as follows:"
        question = ". Please assess the likelihood of this user being a fraudster.Just answer with 'High risk' or 'Low risk'."

        for user, interactions in tqdm(self.user_interaction.items()):
            if user >= self.n_users - self.n_inject_user:
                continue
            detection_text = rs_scenario
            for item in interactions[:-2]:
                detection_text += f" (Item title: {item_features[item]});"
            detection_text = detection_text[:-1] + question
            completion = client.chat.completions.create(
                model="deepseek-v3",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
                messages=[
                {'role': 'user', 'content': detection_text}
                ]
            )
            answer=completion.choices[0].message.content
            detection_emb = LLM_model.general_LLM(answer)
            user_detections[user] = torch.mean(detection_emb, dim=0)

        return user_detections
    
    def _load_detection_inj(self, item_features):
        inj_user_detections = {}
        glo = globals()
        LLM_model = glo[f"{self.LLM}"](self.config)
        rs_scenario = f"In {self.config['recommendation_scenario']}, a user's interaction sequence is as follows:"
        question = ". Please assess the likelihood of this user being a fraudster.Just answer with 'High risk' or 'Low risk'."

        for user, interactions in tqdm(self.user_interaction.items()):
            if user < self.n_users - self.n_inject_user:
                continue
            detection_text = rs_scenario
            for item in interactions[:-2]:
                detection_text += f" (Item title: {item_features[item]});"
            detection_text = detection_text[:-1] + question
            completion = client.chat.completions.create(
                model="deepseek-v3",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
                messages=[
                {'role': 'user', 'content': detection_text}
                ]
            )
            answer=completion.choices[0].message.content
            detection_emb = LLM_model.general_LLM(answer) #[tokens_len,5120]
            inj_user_detections[user] = torch.mean(detection_emb, dim=0) #[5120]

        return inj_user_detections

    def _load_data(self):
        with open("{}/item_dict.json".format(self.path), 'r') as f:
            item_features = json.load(f)
        item_features = {int(k): v for k, v in item_features.items()}
        self.item_id = {k: id+1 for id, k in enumerate(item_features.keys())}
        self.n_items = len(item_features)
        if not os.path.exists(f'{self.path}/pre_processing/{self.LLM}/item_features.pickle') and self.config["use_LLM"]:
            item_features = self._item_fatures_LLM(item_features)
            if not os.path.exists(f'{self.path}/pre_processing/{self.LLM}/'): os.makedirs(f'{self.path}/pre_processing/{self.LLM}/', exist_ok=True)
            with open(f'{self.path}/pre_processing/{self.LLM}/item_features.pickle', 'wb') as f:
                pickle.dump(item_features, f)
        elif os.path.exists(f'{self.path}/pre_processing/{self.LLM}/item_features.pickle') and self.config["use_LLM"]:
            with open(f'{self.path}/pre_processing/{self.LLM}/item_features.pickle', 'rb') as f:
                item_features = pickle.load(f)
                item_features = {int(k): v.to(self.config["device"]) for k, v in item_features.items()}
        self.item_features = {self.item_id[k] : v for k, v in item_features.items()}
        self.item_features[0] = "" if not self.config["use_LLM"] else torch.zeros(self.config["LLM_size"]).to(self.config["device"])
        

        with open("{}/user_dict.json".format(self.path), 'r') as f:
            user_interactions = json.load(f)
        user_interactions = {int(k): v for k, v in user_interactions.items()}
        
        user_interaction_filter = {}
        for user, interaction in user_interactions.items():
            interaction = interaction[-(self.max_len+3):]
            if len(interaction) != len(set(interaction)):   #去除有重复的
                continue
            if not all(int(item) in self.item_id for item in interaction):
                print(f"skip User{user}")
                continue
            interaction = [self.item_id[int(item)] for item in interaction]
            user_interaction_filter[user] = interaction
        
        self.n_users = len(user_interaction_filter)
        self.n_inject_user = 0
        self.user_id = {k: id for id, k in enumerate(user_interaction_filter.keys())}   #user0:0,user1:1...

        if self.config["inject_user"]:
            with open("{}/inject_user_{}.json".format(self.path, self.config["inject_persent"]), 'r') as f:#读取攻击类型为inject_persent的文件
                inject_data = json.load(f)
            inject_user_interactions, target_items = inject_data["user_data"], inject_data["target_item"]
            self.target_item = [self.item_id[item] for item in target_items]
            for user, interaction in inject_user_interactions.items():
                interaction = interaction[-(self.max_len+3):]
                if len(interaction) != len(set(interaction)):
                    continue
                if not all(int(item) in self.item_id for item in interaction):
                    print(f"skip User{user}")
                    continue
                interaction = [self.item_id[int(item)] for item in interaction]
                self.user_id[f"inj_{user}"] = self.n_inject_user + self.n_users
                self.n_inject_user += 1
                user_interaction_filter[f"inj_{user}"] = interaction
            self.n_users += self.n_inject_user
            
        self.user_interaction = {self.user_id[k]: v for k, v in user_interaction_filter.items()} #user0:item_interaction->0:item_id(从1开始)...
        self.user_weight = {self.user_id[k]: self.config["user_weight"] for k in user_interaction_filter.keys()}


        if self.has_fake_user:
            with open("{}/fake_user_dict.json".format(self.path), 'r') as f:
                fake_user_interactions = json.load(f)
            self.n_fake_users = len(fake_user_interactions)
            self.fake_user_id = {k: id for id, k in enumerate(fake_user_interactions.keys())}
            self.fake_data   = {}

            for user, interaction in fake_user_interactions.items():
                interaction = interaction[-(self.max_len+3):]
                if not all(int(item) in self.item_id for item in interaction):
                    print(f"skip User{user}")
                    continue
                interaction = [self.item_id[item] for item in interaction]
                self.fake_data[self.fake_user_id[user]] = interaction

            with open("{}/item_dict.json".format(self.path), 'r') as f:
                item_features = json.load(f)
            item_features = {int(k): v for k, v in item_features.items()}
            item_features = {self.item_id[k] : v for k, v in item_features.items()} #不同点 item_id从1开始
            if not os.path.exists(f'{self.path}/pre_processing/{self.LLM}/detections.pickle'):
                self.user_detections = self._load_detection_users(item_features)
                if not os.path.exists(f'{self.path}/pre_processing/{self.LLM}/'): os.makedirs(f'{self.path}/pre_processing/{self.LLM}/', exist_ok=True)
                with open(f'{self.path}/pre_processing/{self.LLM}/detections.pickle', 'wb') as f:
                    pickle.dump(self.user_detections, f)
            else:
                with open(f'{self.path}/pre_processing/{self.LLM}/detections.pickle', 'rb') as f:
                    self.user_detections = pickle.load(f)
            

            if self.config["inject_user"]:
                if not os.path.exists(f'{self.path}/pre_processing/{self.LLM}/detections_{self.config["inject_persent"]}.pickle'):
                    inj_user_detections = self._load_detection_inj(item_features)
                    if not os.path.exists(f'{self.path}/pre_processing/{self.LLM}/'): os.makedirs(f'{self.path}/pre_processing/{self.LLM}/', exist_ok=True)
                    with open(f'{self.path}/pre_processing/{self.LLM}/detections_{self.config["inject_persent"]}.pickle', 'wb') as f:
                        pickle.dump(inj_user_detections, f)
                else:
                    with open(f'{self.path}/pre_processing/{self.LLM}/detections_{self.config["inject_persent"]}.pickle', 'rb') as f:
                        inj_user_detections = pickle.load(f)
                for user, detection_prmopt in inj_user_detections.items():
                    self.user_detections[user] = detection_prmopt
                    
            if not os.path.exists(f'{self.path}/pre_processing/{self.LLM}/detections_fakes.pickle'):
                self.fake_user_detections = self._load_detection_fake_users(item_features)
                if not os.path.exists(f'{self.path}/pre_processing/{self.LLM}/'): os.makedirs(f'{self.path}/pre_processing/{self.LLM}/', exist_ok=True)
                with open(f'{self.path}/pre_processing/{self.LLM}/detections_fakes.pickle', 'wb') as f:
                    pickle.dump(self.fake_user_detections, f)
            else:
                with open(f'{self.path}/pre_processing/{self.LLM}/detections_fakes.pickle', 'rb') as f:
                    self.fake_user_detections = pickle.load(f)


    def _padding_list(self, list):
        emb_list_padded = [([0] * (self.max_len - len(sub_list)+1) + sub_list) for sub_list in list]    #len(emb_list_padded)=max_len+1,向前扩充，前面补0
        mask_list = [([0] * (self.max_len - len(sub_list)+1) + [1] * len(sub_list)) for sub_list in list]   #有交互1，无交互0

        return emb_list_padded, mask_list

    def get_train_batch(self, idx):
        raise NotImplementedError

    def get_val_batch(self, idx):
        raise NotImplementedError

    def get_test_batch(self, idx):
        raise NotImplementedError

    def get_labels_emb(self, id_list=None):
        if id_list is None:
            return torch.stack(list(self.item_features.values()))
        else:
            return torch.stack([self.item_features[id].to(self.config["device"]) for id in id_list])
    
    def get_interaction_emb(self, id_list):
        return torch.vstack([self.item_features[id] for id in id_list])

    def get_fake_user_batch(self, size=512, neg_idx=None, final_test=0, fk_idx=None, inj_idx=None): #len(neg_idx)=509,len(fk_idx)=58
        if final_test != 0:
            detection_list = [self.fake_user_detections[i] for i in fk_idx]
            org_interaction = [self.fake_data[i][:-2] for i in fk_idx]
            fd_label = [1 for _ in fk_idx]
            if self.config["inject_user"]:
                detection_list += [self.user_detections[i] for i in inj_idx]
                org_interaction += [self.user_interaction[i][:-2] for i in inj_idx]
                fd_label += [1 for _ in inj_idx]
            detection_list += [self.user_detections[i] for i in neg_idx]
            org_interaction += [self.user_interaction[i][:-2] for i in neg_idx]
            fd_label += [0 for _ in neg_idx]
            interaction_list, mask_list = self._padding_list(org_interaction)
        else:
            detection_list = [self.fake_user_detections[i] for i in fk_idx] + [self.user_detections[i] for i in neg_idx]    #[567(509+58),LLM_size:5120] LLM对user是否为欺诈者的预测
            fd_label = [1 for _ in fk_idx] + [0 for _ in neg_idx]
            interaction_list, mask_list = self._padding_list([self.fake_data[i][:-2] for i in fk_idx] + [self.user_interaction[i][:-2] for i in neg_idx])   #[567,max_len:51]
        if self.config["use_LLM"]:
            interaction_list = torch.stack([self.get_interaction_emb(interaction) for interaction in interaction_list]).float() #torch.Size([567, 51, 5120]) user interaction的item编码
        if detection_list is None:
            detection_list=[1]
            print("detection_list is None")
        return interaction_list, mask_list, torch.tensor(fd_label), torch.stack(detection_list).float().to(self.config["device"])
    
    def add_weight(self):
        for user in self.user_weight.keys():
            self.user_weight[user] += 1

    def update_weight(self, neg_idx):#Equation 17
        total_reduction = len(neg_idx)

        for id in neg_idx:
            self.user_weight[id] -= 1

        num_others = len(self.user_weight) - len(neg_idx)
        increase_per_other = total_reduction / num_others

        for key in self.user_weight:
            if key not in neg_idx:
                self.user_weight[key] += increase_per_other

    
    def get_weight(self, idx):
        idx = idx.squeeze().tolist()
        
        weight = []
        for id in idx:
            # weight.append(self.user_weight[id])
            w = torch.sigmoid(torch.tensor(self.user_weight[id])) / torch.sigmoid(torch.tensor(self.config["user_weight"]))
            w = w.item()
            if w > 0.5 and w < 1.0:
                weight.append(1.0)
            else:
                weight.append(w)
        return torch.tensor(weight).unsqueeze(1).to(self.config["device"])

    @staticmethod
    def collate_fn(samples):
        return samples
    
class AlphaDataset(BasicDataset):
    def __init__(self, path, LLM, LLM_config, has_fake_user=False, max_len=1000) -> None:
        super().__init__(path, LLM, LLM_config, has_fake_user, max_len)

        self._load_data()
    
    def get_train_batch(self, idx):
        idx = idx.squeeze().tolist() #torch->list
        whole_item_set = set(self.item_features.keys())
        all_interaction_list = []
        neg_interaction_list = []
        for id in idx:
            interaction_list = self.user_interaction[id]    
            interaction_set = set(interaction_list)
            all_interaction_list.append(interaction_list[:-3])  #真实user与item的交互数据，dim0=bs=512
            neg_interaction_list.append(random.sample(list(whole_item_set - interaction_set), len(interaction_list[:-3])))    #随机不重复采样
        label_list = [self.user_interaction[i][-3] for i in idx]
        all_interaction_list, mask_list = self._padding_list(all_interaction_list)
        neg_interaction_list, _ = self._padding_list(neg_interaction_list)  #两者mask相同
        
        return all_interaction_list, mask_list, neg_interaction_list,label_list #shape=torch.Size([batch_size:512, max_len+1:51, LLM_size:5120]);mask_list.shape=[512,51]
    
    def get_val_batch(self, idx):
        idx = idx.squeeze().tolist()
        label_list = [self.user_interaction[i][-2] for i in idx] 
        interaction_list, mask_list = self._padding_list([self.user_interaction[i][-(self.max_len+2):-2] for i in idx])#[512,51]      
        return interaction_list, mask_list, label_list

    def get_test_batch(self, idx):
        old_id = idx.squeeze().tolist()
        idx = []
        for id in old_id:
            if id < self.n_users - self.n_inject_user:
                idx.append(id)
        if len(idx) == 0:
            return [], [], [], []
        label_list = [self.user_interaction[i][-1] for i in idx]
        interaction_list, mask_list = self._padding_list([self.user_interaction[i][-(self.max_len+1):-1] for i in idx])
        return interaction_list, mask_list, label_list

class SASDataset(BasicDataset):
    def __init__(self, path, LLM, LLM_config, has_fake_user=False, max_len=1000) -> None:
        super().__init__(path, LLM, LLM_config, has_fake_user, max_len)

        self._load_data()
    
    def get_train_batch(self, idx):
        idx = idx.squeeze().tolist() #torch->list
        whole_item_set = set(self.item_features.keys())
        all_interaction_list = []
        neg_interaction_list = []
        for id in idx:
            interaction_list = self.user_interaction[id]    
            interaction_set = set(interaction_list)
            all_interaction_list.append(interaction_list[:-2])  #真实user与item的交互数据，dim0=bs=512
            neg_interaction_list.append(random.sample(list(whole_item_set - interaction_set), len(interaction_list[:-2])))    #随机不重复采样

        all_interaction_list, mask_list = self._padding_list(all_interaction_list)
        neg_interaction_list, _ = self._padding_list(neg_interaction_list)  #两者mask相同

        if self.config["use_LLM"]:
            all_interaction_list = torch.stack([self.get_interaction_emb(interaction) for interaction in all_interaction_list]).float()
            neg_interaction_list = torch.stack([self.get_interaction_emb(interaction) for interaction in neg_interaction_list]).float()
        
        return all_interaction_list, mask_list, neg_interaction_list #shape=torch.Size([batch_size:512, max_len+1:51, LLM_size:5120]);mask_list.shape=[512,51]

    def get_val_batch(self, idx):
        idx = idx.squeeze().tolist()
        label_list = [self.user_interaction[i][-2] for i in idx] 
        interaction_list, mask_list = self._padding_list([self.user_interaction[i][-(self.max_len+2):-2] for i in idx])#[512,51]

        interaction_list_emb = interaction_list
        if self.config["use_LLM"]:
            interaction_list_emb = torch.stack([self.get_interaction_emb(interaction) for interaction in interaction_list]).float()#[512,51,5120]
        
        return interaction_list, interaction_list_emb, mask_list, label_list

    def get_test_batch(self, idx):
        old_id = idx.squeeze().tolist()
        idx = []
        for id in old_id:
            if id < self.n_users - self.n_inject_user:
                idx.append(id)
        if len(idx) == 0:
            return [], [], [], []
        label_list = [self.user_interaction[i][-1] for i in idx]
        interaction_list, mask_list = self._padding_list([self.user_interaction[i][-(self.max_len+1):-1] for i in idx])

        interaction_list_emb = interaction_list
        if self.config["use_LLM"]:
            interaction_list_emb = torch.stack([self.get_interaction_emb(interaction) for interaction in interaction_list]).float()
        
        return interaction_list, interaction_list_emb, mask_list, label_list


class FMLPDataset(SASDataset):
    def __init__(self, path, LLM, LLM_config, has_fake_user=False, max_len=1000) -> None:
        super().__init__(path, LLM, LLM_config, has_fake_user, max_len)


class GRU4recDataset(SASDataset):
    def __init__(self, path, LLM, LLM_config, has_fake_user=False, max_len=1000) -> None:
        super().__init__(path, LLM, LLM_config, has_fake_user, max_len)
