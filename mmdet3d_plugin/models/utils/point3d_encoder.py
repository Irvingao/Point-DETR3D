import copy
import math
import torch
import torch.nn as nn

from projects.mmdet3d_plugin.models.utils.point_encoder import PointEncoderV2

class FixedPointEncoderV2(nn.Module):
    def __init__(self,num_classes,num_feats,scale=None, fix_label_grad=False, with_wlh=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_feats = num_feats
        self.temperature = 10000
        self.label_embed = nn.Embedding(num_classes, 256)
        nn.init.uniform_(self.label_embed.weight)
        self.fix_label_grad = fix_label_grad
        
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        
        # 需要保证pos3d_dim为偶数
        self.pos3d_dim = 256 // 3
        if self.pos3d_dim % 2 == 1:
            self.pos3d_dim -= 1
        self.pad_dim = 256 - self.pos3d_dim * 3
        
        self.with_wlh = with_wlh
        if self.with_wlh:
            self.pos3d_dim = 256 // 6
            if self.pos3d_dim % 2 == 1:
                self.pos3d_dim -= 1
            self.pad_dim = 256 - self.pos3d_dim * 6

            self.gtlabels2names = {-1: 'car',
                0: 'car', 1: 'truck', 2: 'construction_vehicle', 
                3: 'bus', 4: 'trailer', 5: 'barrier', 6: 'motorcycle', 
                7: 'bicycle', 8: 'pedestrian', 9: 'traffic_cone'}

            # [w_mean, w_std, l_mean, l_std, h_mean, h_std]
            self.dict_wlh = {
                'car': [1.96, 0.19, 4.63, 0.47, 1.74, 0.25], 
                'truck': [2.52, 0.45, 6.94, 2.11, 2.85, 0.84], 
                'construction_vehicle': [2.82, 1.06, 6.56, 3.17, 3.2, 0.94], 
                'bus': [2.95, 0.32, 11.19, 2.06, 3.49, 0.49], 
                'trailer': [2.92, 0.55, 12.28, 4.6, 3.87, 0.77], 
                'barrier': [2.51, 0.62, 0.5, 0.17, 0.99, 0.15], 
                'motorcycle': [0.77, 0.16, 2.11, 0.31, 1.46, 0.23], 
                'bicycle': [0.61, 0.16, 1.7, 0.25, 1.3, 0.35], 
                'pedestrian': [0.67, 0.14, 0.73, 0.19, 1.77, 0.19], 
                'traffic_cone': [0.41, 0.14, 0.42, 0.15, 1.08, 0.27]}
            
            

    def calc_emb(self, normed_coord):

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

    def calc_emb_with_wlh(self, normed_coord, labels):

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
            labels = [labels]
        # 这里可能是一个深浅拷贝的问题？
        labels = copy.deepcopy(labels)
        coord = copy.deepcopy(point_coord)

        batch_size = len(coord)
        all_embeddings = []
        positive_num = [coord[idx].size(0) for idx in range(batch_size)]
        # padding_num = [300 - positive_num[idx] for idx in range(batch_size)]
        padding_num = [max(positive_num) - positive_num[idx] for idx in range(batch_size)]

        for idx in range(batch_size):
            # 防止test时遇到空GT报错
            try:
                label_embedding = self.label_embed.weight[labels[idx].long()]
                # print(f"label_embedding: {label_embedding.shape}")
            except IndexError:
                # 如果没有GT box
                # print('-------------------------')
                # print(labels)
                # print(point_coord)
                # print(len(labels))
                # print(len(point_coord))
                # print('-------------------------')
                # if len(labels) == 0:
                label_embedding = self.label_embed.weight[0]
                coord = [coord[0].new_tensor([[0.,0.,0.]]) for i in range(batch_size)]
                
                positive_num = [coord[idx].size(0) for idx in range(batch_size)]
                # print('||||||||||||||||||||||||||')
                # print(label_embedding.shape)    # torch.Size([256])
                # print(coord)
                # print('||||||||||||||||||||||||||')
            
            if self.fix_label_grad:
                label_embedding.requires_grad = False
            
            # label_embedding = self.label_embed.weight[labels[idx].long()]
            
            # 将原始的坐标归一化的0-1尺度   添加这里之后初始的loss直接从115下降到62
            coord[idx][..., 0:1] = (coord[idx][..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
            coord[idx][..., 1:2] = (coord[idx][..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
            coord[idx][..., 2:3] = (coord[idx][..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

            # print(f"coords[0]: {coord[0].shape}")   # [n, 3]
            
            if self.with_wlh:
                position_embedding = self.calc_emb_with_wlh(coord[idx], labels[idx].long())
            else:
                position_embedding = self.calc_emb(coord[idx])

            position_embedding = position_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)
            # self.adapt_pos3d.to(position_embedding.device)
            # position_embedding = self.adapt_pos3d(position_embedding)
            position_embedding = position_embedding.squeeze().squeeze()
            if padding_num[idx] == 0:
                # 处理 为1时的异常
                if len(position_embedding.shape) == 1 and max(positive_num) == 1:   
                    position_embedding = position_embedding.unsqueeze(0)
                query_embedding =  label_embedding.to(position_embedding.device) + position_embedding
                query_pos = nn.Embedding(max(positive_num), self.num_feats * 2)
                # print(f"positive_num: {positive_num}")   # [n]
                # print(f"position_embedding: {position_embedding.shape}")   # [n, 256]
                # print(f"query_pos: {query_pos.weight.shape}")   # [n, 256]
                # print(f"query_embedding: {query_embedding.shape}")  # [n, 256]
                query_embedding = torch.cat((query_pos.weight.to(query_embedding.device), query_embedding), dim=1)
                all_embeddings.append(query_embedding)
                continue
            else:
                position_embedding = torch.cat([position_embedding, 
                    nn.Embedding(padding_num[idx], 256).weight.to(position_embedding.device)],dim=0)
                # print("labels:",labels)
                # print("label_embedding:",label_embedding.size())
                # print("padding_embedding:", nn.Embedding(padding_num[idx], 256).weight.size())
                if label_embedding.size()[0]==256:
                    label_embedding = label_embedding.unsqueeze(0)
                    # label_embedding = torch.cat([label_embedding.to(position_embedding.device),nn.Embedding(299, 256).weight.to(position_embedding.device)],dim=0)
                    label_embedding = torch.cat([label_embedding.to(position_embedding.device),
                                                 nn.Embedding(padding_num[idx], 256).weight.to(position_embedding.device)], dim=0)
                else:
                    label_embedding = torch.cat([label_embedding.to(position_embedding.device),nn.Embedding(padding_num[idx], 256).weight.to(position_embedding.device)], dim=0)

                query_embedding = label_embedding + position_embedding
                # 为了和后面对齐，所以这里再cancat一个[num_point,256]
                # num_points = query_embedding.shape[0]
                # query_pos = nn.Embedding(300, self.num_feats*2)
                query_pos = nn.Embedding(max(positive_num), self.num_feats * 2)
                query_embedding = torch.cat((query_pos.weight.to(query_embedding.device),query_embedding),dim=1)
                all_embeddings.append(query_embedding)

        all_embeddings = torch.stack(all_embeddings,dim=0)
        # print(f"all_embeddings: {all_embeddings.shape}")
        
        if all_embeddings.size(1) == 0: # 空GT
            all_embeddings = all_embeddings.new_zeros((all_embeddings.size(0),1,all_embeddings.size(2)))
        
        return all_embeddings


class Point3DEncoder(FixedPointEncoderV2):
    def __init__(self, 
                 num_classes,
                 num_feats,
                 scale=None, 
                 with_wlh=False, 
                 fix_label_grad=False):
        super().__init__(num_classes=num_classes, num_feats=num_feats,
            scale=scale, with_wlh=with_wlh, fix_label_grad=fix_label_grad)
        
    def forward(self, point_coord, labels, pc_range):
        if isinstance(labels, torch.Tensor):
            labels = [labels]
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
            
            if self.fix_label_grad:
                # label_embedding.requires_grad = False
                label_embedding = label_embedding.detach()
            
            # 将原始的坐标归一化的0-1尺度   添加这里之后初始的loss直接从115下降到62
            coord[idx][..., 0:1] = (coord[idx][..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
            coord[idx][..., 1:2] = (coord[idx][..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
            coord[idx][..., 2:3] = (coord[idx][..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

            if self.with_wlh:
                position_embedding = self.calc_emb_with_wlh(coord[idx], labels[idx].long())
            else:
                position_embedding = self.calc_emb(coord[idx])

            position_embedding = position_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)
            position_embedding = position_embedding.squeeze().squeeze()
            
            # 处理 为1时的异常
            if len(position_embedding.shape) == 1 and max(positive_num) == 1:   
                position_embedding = position_embedding.unsqueeze(0)
            query_embedding =  label_embedding.to(position_embedding.device) + position_embedding
            query_pos = position_embedding
            query_embedding = torch.cat((query_pos,query_embedding), dim=1)
            all_embeddings.append(query_embedding)
            continue
            
        all_embeddings = torch.stack(all_embeddings,dim=0)
        # print(f"all_embeddings: {all_embeddings.shape}")
        
        if all_embeddings.size(1) == 0: # 空GT
            all_embeddings = all_embeddings.new_zeros((all_embeddings.size(0),1,all_embeddings.size(2)))
        
        return all_embeddings   # [query_pos, query_embedding]
    

class Point3DEncoderV2(FixedPointEncoderV2):
    def __init__(self, 
                 num_classes,
                 num_feats,
                 learnable_query=False,
                 learnable_query_pos=False,
                 scale=None, 
                 with_wlh=False, 
                 fix_label_grad=True):
        super().__init__(num_classes=num_classes, num_feats=num_feats,
            scale=scale, with_wlh=with_wlh, fix_label_grad=fix_label_grad)
        
        self.learnable_query = learnable_query
        if self.learnable_query:
            # class wise learnable
            self.query_embed = nn.Embedding(num_classes, 256)
        self.learnable_query_pos = learnable_query_pos
        if self.learnable_query_pos:
            # class wise learnable pos
            self.query_pos_embed = nn.Embedding(num_classes, 256)
            
        
        if self.fix_label_grad:
            # 固定embedding
            self.label_embed.weight.requires_grad = False
        
    def forward(self, point_coord, labels, pc_range):
        if isinstance(labels, torch.Tensor):
            labels = [labels]
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
                if self.learnable_query:
                    query_embed = self.query_embed.weight[labels[idx].long()]
                if self.learnable_query_pos:
                    query_pos_embed = self.query_pos_embed.weight[labels[idx].long()]
                
            except IndexError:
                # 如果没有GT box
                label_embedding = self.label_embed.weight[0]
                coord = [coord[0].new_tensor([[0.,0.,0.]]) for i in range(batch_size)]
                
                positive_num = [coord[idx].size(0) for idx in range(batch_size)]
                # print(label_embedding.shape)    # torch.Size([256])
                if self.learnable_query:
                    query_embed = self.query_embed.weight[0]
                if self.learnable_query_pos:
                    query_pos_embed = self.query_pos_embed.weight[0]
            
            # 将原始的坐标归一化的0-1尺度   添加这里之后初始的loss直接从115下降到62
            coord[idx][..., 0:1] = (coord[idx][..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
            coord[idx][..., 1:2] = (coord[idx][..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
            coord[idx][..., 2:3] = (coord[idx][..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

            if self.with_wlh:
                position_embedding = self.calc_emb_with_wlh(coord[idx], labels[idx].long())
            else:
                position_embedding = self.calc_emb(coord[idx])

            position_embedding = position_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)
            position_embedding = position_embedding.squeeze().squeeze()
            
            # 处理 为1时的异常
            if len(position_embedding.shape) == 1 and max(positive_num) == 1:   
                position_embedding = position_embedding.unsqueeze(0)
            query_embedding =  label_embedding.to(position_embedding.device) + position_embedding
            query_pos = position_embedding
            if self.learnable_query:
                query_embedding = query_embedding + query_embed.to(position_embedding.device)
            if self.learnable_query_pos:
                query_pos = query_pos + query_pos_embed.to(position_embedding.device)
            query_embedding = torch.cat((query_pos,query_embedding), dim=1)
            all_embeddings.append(query_embedding)
            continue
            
        all_embeddings = torch.stack(all_embeddings,dim=0)
        # print(f"all_embeddings: {all_embeddings.shape}")
        
        if all_embeddings.size(1) == 0: # 空GT
            all_embeddings = all_embeddings.new_zeros((all_embeddings.size(0),1,all_embeddings.size(2)))
        
        return all_embeddings   # [query_pos, query_embedding]
