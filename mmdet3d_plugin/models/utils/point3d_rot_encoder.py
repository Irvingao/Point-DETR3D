import copy
import math
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init

from projects.mmdet3d_plugin.datasets.nuscenes_utils.statistics_data import gtlabels2names, dict_wlh



class RotPoint3DEncoder(nn.Module):
    def __init__(self, 
                 num_classes,
                 embed_dims=256,
                 alpha=1,
                 pos='position_embedding',
                 with_wlh=True,
                 rot_label=False,
                 rot_encoding='V2',
                 query_proj=None,
                 learnable_label=True,
                 scale=None):
        super().__init__()
        self.num_classes = num_classes
        self.temperature = 10000
        self.alpha = alpha
        self.pos = pos
        
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        
        num_params = 6 if with_wlh else 3
        # 需要保证pos3d_dim为偶数
        self.pos3d_dim = embed_dims // num_params
        if self.pos3d_dim % 2 == 1:
            self.pos3d_dim -= 1
        self.pad_dim = embed_dims - self.pos3d_dim * num_params
        
        self.with_wlh = with_wlh
        self.gtlabels2names = gtlabels2names
        self.dict_wlh = dict_wlh
        
        # label_embed
        self.rot_label = rot_label
        self.rot_encoding = rot_encoding
        assert self.rot_encoding in ['V1', 'V2'], "see code here"
        
        self.query_proj = query_proj
        if query_proj: 
            self.query_proj = nn.Linear(embed_dims, embed_dims)
        
        self.learnable_label = learnable_label
        self.label_embed = nn.Embedding(num_classes, embed_dims)
    
        self.init_weight()    
        '''
        # 需要保证pos3d_dim为偶数
        self.pos3d_dim = embed_dims // 3
        if self.pos3d_dim % 2 == 1:
            self.pos3d_dim -= 1
        self.pad_dim = embed_dims - self.pos3d_dim * 3
        
        if self.with_wlh:
            self.pos3d_dim = embed_dims // 6
            if self.pos3d_dim % 2 == 1:
                self.pos3d_dim -= 1
            self.pad_dim = embed_dims - self.pos3d_dim * 6

            self.gtlabels2names = gtlabels2names
            self.dict_wlh = dict_wlh
        '''
    
    def init_weight(self):
        if self.query_proj is not None:
            constant_init(self.query_proj, val=1., bias=0.)
        
        if self.learnable_label:
            if self.rot_label:
                self.rot_label_encoder()
            else:
                nn.init.uniform_(self.label_embed.weight)
        else:
            if self.rot_label:
                # 固定embedding
                self.label_embed.weight.requires_grad = False
                # nn.init.zeros_(self.label_embed.weight).to(torch.float32)
                self.rot_label_encoder()
            else:
                nn.init.uniform_(self.label_embed.weight)
            # 固定embedding
            # self.label_embed.weight.requires_grad = False
    
    def rot_label_encoder(self):
        num_cls, dims = self.label_embed.weight.shape
        
        # 计算正弦编码和余弦编码，这里使用了正弦位置编码
        rot_pos_enc = torch.zeros(num_cls, dims)
        
        if self.rot_encoding == 'V1':
            values = torch.range(1, num_cls) # 生成有序列表
            angles = values * (2 * torch.tensor([math.pi]) / num_cls) # 将列表映射到圆形上
            rot_pos_enc[:, 0::2] = torch.sin(angles.unsqueeze(1) / torch.pow(self.temperature, torch.arange(0, dims, 2, dtype=torch.float32) / dims))
            rot_pos_enc[:, 1::2] = torch.cos(angles.unsqueeze(1) / torch.pow(self.temperature, torch.arange(1, dims, 2, dtype=torch.float32) / dims))
        elif self.rot_encoding == 'V2':
            values = torch.range(0, num_cls)    # 生成有序列表
            angles = values * (2 * torch.tensor([math.pi]) / num_cls)   # 将列表映射到圆形上
            angle_sectors = []
            for i in range(self.num_classes):
                angle_sector = torch.linspace(angles[i], angles[i+1], dims//2, dtype=torch.float32)
                angle_sectors.append(angle_sector)
            angle_sectors = torch.stack(angle_sectors, dim=0)
            # rot_temperature = torch.ones(dims//2, dtype=torch.float32) * alpha
            rot_pos_enc[:, 0::2] = angle_sectors.sin() / self.alpha
            rot_pos_enc[:, 1::2] = angle_sectors.cos() / self.alpha
        else:
            raise ValueError(f"`rot_encoding` doesn't support as `{rot_encoding}`.")
        # (rot_pos_enc < 1).all() and (rot_pos_enc > 0).all() 
        
        self.label_embed.weight.data = rot_pos_enc
        
    def position_encoder(self, normed_coord):

        normed_coord = normed_coord * self.scale
        device = normed_coord.device

        dim_t = torch.arange(self.pos3d_dim, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.pos3d_dim)

        pos_x = normed_coord[:, 0, None] / dim_t  # [N,256/3]
        pos_y = normed_coord[:, 1, None] / dim_t
        pos_z = normed_coord[:, 2, None] / dim_t

        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()), dim=2).flatten(1)
        # 这里的cat顺序参照PETRV2
        pos = torch.cat((pos_y,pos_x, pos_z), dim=-1)
        
        if self.pad_dim > 0:
            pos = torch.cat((pos,torch.zeros_like(pos_x)[:, :self.pad_dim]), dim=-1)
        return pos

    def position_encoder_with_wlh(self, normed_coord, labels):

        normed_coord = normed_coord * self.scale
        device = normed_coord.device

        dim_t = torch.arange(self.pos3d_dim, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.pos3d_dim)

        pos_x = normed_coord[:, 0, None] / dim_t  # [N,256]
        pos_y = normed_coord[:, 1, None] / dim_t
        pos_z = normed_coord[:, 2, None] / dim_t

        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()), dim=2).flatten(1)
        # 这里的cat顺序参照PETRV2
        pos = torch.cat((pos_y,pos_x, pos_z), dim=-1)
        
        labels = labels.cpu().numpy().tolist() 
        pos_w = normed_coord.new_tensor([self.dict_wlh[self.gtlabels2names[label]][0] for label in labels])[:, None] / dim_t # [N,256/6]
        pos_l = normed_coord.new_tensor([self.dict_wlh[self.gtlabels2names[label]][2] for label in labels])[:, None] / dim_t
        pos_h = normed_coord.new_tensor([self.dict_wlh[self.gtlabels2names[label]][4] for label in labels])[:, None] / dim_t

        pos_w = torch.stack((pos_w[:, 0::2].sin(), pos_w[:, 1::2].cos()), dim=2).flatten(1)
        pos_l = torch.stack((pos_l[:, 0::2].sin(), pos_l[:, 1::2].cos()), dim=2).flatten(1)
        pos_h = torch.stack((pos_l[:, 0::2].sin(), pos_l[:, 1::2].cos()), dim=2).flatten(1)
        
        wlh = torch.cat((pos_w, pos_l, pos_h), dim=-1)
        
        pos_wlh = torch.cat((pos, wlh), dim=-1)
        
        if self.pad_dim > 0:
            pos_wlh = torch.cat((pos_wlh, torch.zeros_like(pos_w)[:, :self.pad_dim]), dim=-1)
        
        return pos_wlh    
    
    def forward(self, point_coord, labels, pc_range):
        if isinstance(labels, torch.Tensor):
            labels = [labels]   # when test pipeline
        # 这里可能是一个深浅拷贝的问题？
        labels = copy.deepcopy(labels)
        coord = copy.deepcopy(point_coord)

        batch_size = len(coord)
        assert batch_size == 1
        all_embeddings = []
        positive_num = [coord[idx].size(0) for idx in range(batch_size)]

        for idx in range(batch_size):
            # 防止test时遇到空GT报错
            try:
                label_embedding = self.label_embed.weight[labels[idx].long()]
                
            except IndexError:
                # 如果没有GT box
                label_embedding = self.label_embed.weight[0]
                coord = [coord[0].new_tensor([[0.,0.,0.]]) for i in range(batch_size)]
                
                positive_num = [coord[idx].size(0) for idx in range(batch_size)]
                # print(label_embedding.shape)    # torch.Size([256])
            
            # 将原始的坐标归一化的0-1尺度   添加这里之后初始的loss直接从115下降到62
            coord[idx][..., 0:1] = (coord[idx][..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
            coord[idx][..., 1:2] = (coord[idx][..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
            coord[idx][..., 2:3] = (coord[idx][..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

            if self.with_wlh:
                position_embedding = self.position_encoder_with_wlh(coord[idx], labels[idx].long())
            else:
                position_embedding = self.position_encoder(coord[idx])

            position_embedding = position_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)
            position_embedding = position_embedding.squeeze().squeeze()
            
            # 处理 为1时的异常
            if len(position_embedding.shape) == 1 and max(positive_num) == 1:   
                position_embedding = position_embedding.unsqueeze(0)
            
            query_embedding = label_embedding.to(position_embedding.device) + position_embedding

            if self.pos == 'position_embedding':
                query_pos = position_embedding
            elif self.pos == 'query_embedding':
                query_pos = query_embedding
            else:
                query_pos = torch.zeros_like(query_embedding)
            
            if self.query_proj is not None:
                query_embedding = self.query_proj(query_embedding)

            query_embedding = torch.cat((query_pos, query_embedding), dim=1)
            all_embeddings.append(query_embedding)
            continue
            
        all_embeddings = torch.stack(all_embeddings,dim=0)
        # print(f"all_embeddings: {all_embeddings.shape}")
        
        if all_embeddings.size(1) == 0: # 空GT
            all_embeddings = all_embeddings.new_zeros((all_embeddings.size(0),1,all_embeddings.size(2)))
        
        return all_embeddings   # [query_pos, query_embedding]

'''
class RotPoint3DEncoderV2(RotPoint3DEncoder):
    def __init__(self, 
                 num_classes,
                 num_feats, 
                 with_wlh=False,
                 rot_label=True,
                 learnable_label=False,
                 scale=None):
        super().__init__(num_classes=num_classes, num_feats=num_feats, rot_label=rot_label, 
                         scale=scale, with_wlh=with_wlh, learnable_label=learnable_label)
        
        num_params = 3 if  not with_wlh else 6
        num_params += 1     # classes
        # 需要保证pos3d_dim为偶数
        self.pos3d_dim = 256 // num_params
        if self.pos3d_dim % 2 == 1:
            self.pos3d_dim -= 1
        self.pad_dim = 256 - self.pos3d_dim * num_params

    def rot_label_encoder(self):
        num_cls, dims = self.label_embed.weight.shape
        # 生成有序列表
        values = torch.arange(num_cls)
        # 将列表映射到圆形上
        angles = num_cls * (2 * torch.tensor([math.pi]) / num_cls)
        # 计算正弦编码和余弦编码，这里使用了正弦位置编码
        rot_pos_enc = torch.zeros(num_cls, dims)
        rot_pos_enc[:, 0::2] = torch.sin(angles.unsqueeze(1) / torch.pow(self.temperature, torch.arange(0, dims, 2, dtype=torch.float32) / dims))
        rot_pos_enc[:, 1::2] = torch.cos(angles.unsqueeze(1) / torch.pow(self.temperature, torch.arange(1, dims, 2, dtype=torch.float32) / dims))
        
        self.label_embed.weight.data = rot_pos_enc
'''


# %%
