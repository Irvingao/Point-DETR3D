import copy
import math

import torch
import torch.nn as nn

class PointEncoder2D(nn.Module):
    def __init__(self,num_classes,num_feats=128, pos=None, scale=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_feats = num_feats
        self.pos = pos
        self.temperature = 10000
        self.label_embed = nn.Embedding(num_classes, 256)
        nn.init.uniform_(self.label_embed.weight)

        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def calc_emb(self, normed_coord):
        normed_coord = normed_coord * self.scale
        device = normed_coord.device
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        pos_x = normed_coord[:, 0, None] / dim_t  # NxC
        pos_y = normed_coord[:, 1, None] / dim_t  # NxC
        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos = torch.cat((pos_x,pos_y), dim=-1)
        return pos

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


            position_embedding = self.calc_emb(coord[idx])

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
            query_embedding = torch.cat((query_pos, query_embedding), dim=1)
            all_embeddings.append(query_embedding)
            continue
            
        all_embeddings = torch.stack(all_embeddings,dim=0)
        
        if all_embeddings.size(1) == 0: # 空GT
            all_embeddings = all_embeddings.new_zeros((all_embeddings.size(0),1,all_embeddings.size(2)))
        
        return all_embeddings   # [query_pos, query_embedding]

class PointEncoder(nn.Module):

    def __init__(self,num_classes,num_feats,scale=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_feats = num_feats
        self.temperature = 10000
        # self.query_emb = nn.Embedding(100, 256)
        self.label_embed = nn.Embedding(num_classes, 256)
        nn.init.uniform_(self.label_embed.weight)

        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.num_feats * 3, self.num_feats * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.num_feats * 4, self.num_feats*2, kernel_size=1, stride=1, padding=0),
        )

        if scale is None:
            scale = 2 * math.pi
        self.scale = scale



    def calc_emb(self, normed_coord):

        # normed_coord = normed_coord[0]
        # normed_coord = normed_coord.clamp(0., 1.)
        # normed_coord = normed_coord.clamp(0., 1.) * self.scale
        normed_coord = normed_coord * self.scale
        device = normed_coord.device

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = normed_coord[:, 0, None] / dim_t  # NxC
        pos_y = normed_coord[:, 1, None] / dim_t  # NxC
        pos_z = normed_coord[:, 2, None] / dim_t

        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()), dim=2).flatten(1)

        # 这里的cat顺序参照PETRV2
        pos = torch.cat((pos_y,pos_x, pos_z), dim=-1)
        return pos

    def forward(self, point_coord,labels,pc_range):
        # 这里可能是一个深浅拷贝的问题？
        labels = copy.deepcopy(labels)
        coord = copy.deepcopy(point_coord)

        batch_size = len(coord)
        all_embeddings = []
        for idx in range(batch_size):
            label_embedding = self.label_embed.weight[labels[idx].long()]

        # 将原始的坐标归一化的0-1尺度   添加这里之后初始的loss直接从115下降到62
            coord[idx][..., 0:1] = (coord[idx][..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
            coord[idx][..., 1:2] = (coord[idx][..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
            coord[idx][..., 2:3] = (coord[idx][..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])


            position_embedding = self.calc_emb(coord[idx])

            position_embedding = position_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)
            self.adapt_pos3d.to(position_embedding.device)
            position_embedding = self.adapt_pos3d(position_embedding)
            position_embedding = position_embedding.squeeze().squeeze()

            label_embedding = label_embedding.to(position_embedding.device)
            query_embedding = label_embedding + position_embedding

            # 为了和后面对齐，所以这里再cancat一个[num_point,256]
            # num_points = query_embedding.shape[0]
            # query_pos = nn.Embedding(num_points, self.num_feats*2)
            #
            # query_embedding = torch.cat((query_pos.weight.to(query_embedding.device),query_embedding),dim=1)

            all_embeddings.append(query_embedding)

        return all_embeddings

class PointEncoderV2(nn.Module):

    def __init__(self,num_classes,num_feats,scale=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_feats = num_feats
        self.temperature = 10000
        # self.query_emb = nn.Embedding(100, 256)
        self.label_embed = nn.Embedding(num_classes, 256)
        nn.init.uniform_(self.label_embed.weight)

        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.num_feats * 3, self.num_feats * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.num_feats * 4, self.num_feats*2, kernel_size=1, stride=1, padding=0),
        )

        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def calc_emb(self, normed_coord):

        # normed_coord = normed_coord[0]
        # normed_coord = normed_coord.clamp(0., 1.)
        # normed_coord = normed_coord.clamp(0., 1.) * self.scale
        normed_coord = normed_coord * self.scale
        device = normed_coord.device

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = normed_coord[:, 0, None] / dim_t  # NxC
        pos_y = normed_coord[:, 1, None] / dim_t  # NxC
        pos_z = normed_coord[:, 2, None] / dim_t

        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()), dim=2).flatten(1)

        # 这里的cat顺序参照PETRV2
        pos = torch.cat((pos_y,pos_x, pos_z), dim=-1)
        return pos

    def forward(self, point_coord, labels, pc_range):
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
            
            # label_embedding = self.label_embed.weight[labels[idx].long()]
            
            # 将原始的坐标归一化的0-1尺度   添加这里之后初始的loss直接从115下降到62
            coord[idx][..., 0:1] = (coord[idx][..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
            coord[idx][..., 1:2] = (coord[idx][..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
            coord[idx][..., 2:3] = (coord[idx][..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

            # print(f"coords[0]: {coord[0].shape}")   # [n, 3]
            
            position_embedding = self.calc_emb(coord[idx])

            position_embedding = position_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)
            self.adapt_pos3d.to(position_embedding.device)
            position_embedding = self.adapt_pos3d(position_embedding)
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
                position_embedding = torch.cat([position_embedding,nn.Embedding(padding_num[idx], 256).weight.to(position_embedding.device)],dim=0)
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
        return all_embeddings


class PointEncoderV3(nn.Module):

    def __init__(self,num_classes,num_feats,scale=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_feats = num_feats
        self.temperature = 10000
        # self.query_emb = nn.Embedding(100, 256)
        self.label_embed = nn.Embedding(num_classes, 256)
        nn.init.uniform_(self.label_embed.weight)

        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.num_feats * 3, self.num_feats * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.num_feats * 4, self.num_feats*2, kernel_size=1, stride=1, padding=0),
        )

        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def calc_emb(self, normed_coord):

        # normed_coord = normed_coord[0]
        # normed_coord = normed_coord.clamp(0., 1.)
        # normed_coord = normed_coord.clamp(0., 1.) * self.scale
        normed_coord = normed_coord * self.scale
        device = normed_coord.device

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = normed_coord[:, 0, None] / dim_t  # NxC
        pos_y = normed_coord[:, 1, None] / dim_t  # NxC
        pos_z = normed_coord[:, 2, None] / dim_t

        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()), dim=2).flatten(1)

        # 这里的cat顺序参照PETRV2
        pos = torch.cat((pos_y,pos_x, pos_z), dim=-1)
        return pos


    def forward(self, point_coord,labels,pc_range):

        labels = copy.deepcopy(labels)
        coord = copy.deepcopy(point_coord)

        batch_size = len(coord)
        all_embeddings = []
        positive_num = [coord[idx].size(0) for idx in range(batch_size)]
        padding_num = [max(positive_num)- positive_num[idx] for idx in range(batch_size)]

        for idx in range(batch_size):
            label_embedding = self.label_embed.weight[labels[idx].long()]
            # 将原始的坐标归一化的0-1尺度
            coord[idx][..., 0:1] = (coord[idx][..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
            coord[idx][..., 1:2] = (coord[idx][..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
            coord[idx][..., 2:3] = (coord[idx][..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

            position_embedding = self.calc_emb(coord[idx])

            position_embedding = position_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)
            self.adapt_pos3d.to(position_embedding.device)
            position_embedding = self.adapt_pos3d(position_embedding)
            position_embedding = position_embedding.squeeze().squeeze()
            # 先将其进行padding
            if padding_num[idx] == 0:
                query_embedding = label_embedding.to(position_embedding.device) + position_embedding
                all_embeddings.append(query_embedding)
                continue
            else:
                if position_embedding.size()[0]==256:
                    position_embedding = position_embedding.unsqueeze(0)
                    position_embedding = torch.cat([position_embedding.to(position_embedding.device),
                                                 nn.Embedding(positive_num[idx]-1, 256).weight.to(position_embedding.device)], dim=0)
                else:
                    position_embedding = torch.cat([position_embedding,nn.Embedding(padding_num[idx], 256).weight.to(position_embedding.device)],dim=0)
                # print("labels:",labels)
                # print("label_embedding:",label_embedding.size())
                # print("padding_embedding:", nn.Embedding(padding_num[idx], 256).weight.size())
                if label_embedding.size()[0]==256:
                    label_embedding = label_embedding.unsqueeze(0)
                    label_embedding = torch.cat([label_embedding.to(position_embedding.device),nn.Embedding(positive_num[idx]-1, 256).weight.to(position_embedding.device)],dim=0)
                else:
                    label_embedding = torch.cat([label_embedding.to(position_embedding.device),nn.Embedding(padding_num[idx], 256).weight.to(position_embedding.device)], dim=0)

            query_embedding = label_embedding + position_embedding
            all_embeddings.append(query_embedding)

        all_embeddings = torch.stack(all_embeddings,dim=0)

        return all_embeddings


#
# class PointEncoderV3(nn.Module):
#
#     def __init__(self, num_classes, num_feats, scale=None):
#         super().__init__()
#         self.num_classes = num_classes
#         self.num_feats = num_feats
#         self.temperature = 10000
#         # self.query_emb = nn.Embedding(100, 256)
#         self.label_embed = nn.Embedding(num_classes, 256)
#         nn.init.uniform_(self.label_embed.weight)
#
#         self.adapt_pos3d = nn.Sequential(
#             nn.Conv2d(self.num_feats * 3, self.num_feats * 4, kernel_size=1, stride=1, padding=0),
#             nn.ReLU(),
#             nn.Conv2d(self.num_feats * 4, self.num_feats * 2, kernel_size=1, stride=1, padding=0),
#         )
#
#         if scale is None:
#             scale = 2 * math.pi
#         self.scale = scale
#
#     def calc_emb(self, normed_coord):
#
#         # normed_coord = normed_coord[0]
#         # normed_coord = normed_coord.clamp(0., 1.)
#         # normed_coord = normed_coord.clamp(0., 1.) * self.scale
#         normed_coord = normed_coord * self.scale
#         device = normed_coord.device
#
#         dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=device)
#         dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
#
#         pos_x = normed_coord[:, 0, None] / dim_t  # NxC
#         pos_y = normed_coord[:, 1, None] / dim_t  # NxC
#         pos_z = normed_coord[:, 2, None] / dim_t
#
#         pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
#         pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
#         pos_z = torch.stack((pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()), dim=2).flatten(1)
#
#         # 这里的cat顺序参照PETRV2
#         pos = torch.cat((pos_y, pos_x, pos_z), dim=-1)
#         return pos
#
#     def forward(self, point_coord, labels, pc_range):
#
#         labels = copy.deepcopy(labels)
#         coord = copy.deepcopy(point_coord)
#
#         batch_size = len(coord)
#         all_embeddings = []
#         positive_num = [coord[idx].size(0) for idx in range(batch_size)]
#         padding_num = [max(positive_num) - positive_num[idx] for idx in range(batch_size)]
#
#         for idx in range(batch_size):
#             if len(labels) == 0:
#                 label_embedding = self.label_embed.weight[0]
#             else:
#                 label_embedding = self.label_embed.weight[labels[idx].long()]
#             # label_embedding = self.label_embed.weight[labels[idx].long()]
#             # 将原始的坐标归一化的0-1尺度
#             coord[idx][..., 0:1] = (coord[idx][..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
#             coord[idx][..., 1:2] = (coord[idx][..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
#             coord[idx][..., 2:3] = (coord[idx][..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])
#
#             position_embedding = self.calc_emb(coord[idx])
#
#             position_embedding = position_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)
#             self.adapt_pos3d.to(position_embedding.device)
#             position_embedding = self.adapt_pos3d(position_embedding)
#             position_embedding = position_embedding.squeeze().squeeze()
#             # 先将其进行padding
#             if padding_num[idx] == 0:
#                 query_embedding = label_embedding.to(position_embedding.device) + position_embedding
#                 all_embeddings.append(query_embedding)
#                 continue
#             else:
#                 if position_embedding.size()[0] == 256:
#                     position_embedding = position_embedding.unsqueeze(0)
#                     position_embedding = torch.cat(
#                         [position_embedding, nn.Embedding(padding_num[idx], 256).weight.to(position_embedding.device)],
#                         dim=0)
#                 else:
#                     position_embedding = torch.cat(
#                         [position_embedding, nn.Embedding(padding_num[idx], 256).weight.to(position_embedding.device)],
#                         dim=0)
#                 # print("labels:",labels)
#                 # print("label_embedding:",label_embedding.size())
#                 # print("padding_embedding:", nn.Embedding(padding_num[idx], 256).weight.size())
#                 if label_embedding.size()[0] == 256:
#                     label_embedding = label_embedding.unsqueeze(0)
#                     label_embedding = torch.cat([label_embedding.to(position_embedding.device),
#                                                  nn.Embedding(padding_num[idx], 256).weight.to(
#                                                      position_embedding.device)], dim=0)
#                 else:
#                     label_embedding = torch.cat([label_embedding.to(position_embedding.device),
#                                                  nn.Embedding(padding_num[idx], 256).weight.to(
#                                                      position_embedding.device)], dim=0)
#
#             query_embedding = label_embedding + position_embedding
#             all_embeddings.append(query_embedding)
#
#         all_embeddings = torch.stack(all_embeddings, dim=0)
#
#         return all_embeddings
#

class Hybrid_PointEncoder(nn.Module):

    def __init__(self,num_classes,num_feats,scale=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_feats = num_feats
        self.temperature = 10000
        # self.query_emb = nn.Embedding(100, 256)
        self.label_embed = nn.Embedding(num_classes, 256)
        nn.init.uniform_(self.label_embed.weight)

        self.adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.num_feats * 3, self.num_feats * 4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.num_feats * 4, self.num_feats*2, kernel_size=1, stride=1, padding=0),
        )

        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def calc_emb(self, normed_coord):

        normed_coord = normed_coord * self.scale
        device = normed_coord.device

        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)

        pos_x = normed_coord[:, 0, None] / dim_t  # NxC
        pos_y = normed_coord[:, 1, None] / dim_t  # NxC
        pos_z = normed_coord[:, 2, None] / dim_t

        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
        pos_z = torch.stack((pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()), dim=2).flatten(1)

        # 这里的cat顺序参照PETRV2
        pos = torch.cat((pos_y,pos_x, pos_z), dim=-1)
        return pos

    def one2many_query_init(self,point_coord,labels,pc_range,all_embeddings):

        '''
        return one2many_query[list] : 表示每个batch的one2many query
        '''
        batch_size = len(point_coord)
        positive_num = [point_coord[idx].size(0) for idx in range(batch_size)]
        one2many_num = 300 - max(positive_num)
        for idx in range(batch_size):
            all_embeddings[idx] = torch.cat((all_embeddings[idx],nn.Embedding(one2many_num, 256).weight.to(all_embeddings[idx].device)), dim=0)
        return all_embeddings




    def forward(self, point_coord,labels,pc_range):

        labels = copy.deepcopy(labels)
        coord = copy.deepcopy(point_coord)

        batch_size = len(coord)
        all_embeddings = []
        positive_num = [coord[idx].size(0) for idx in range(batch_size)]
        padding_num = [max(positive_num)- positive_num[idx] for idx in range(batch_size)]

        for idx in range(batch_size):
            label_embedding = self.label_embed.weight[labels[idx].long()]
            # 将原始的坐标归一化的0-1尺度
            coord[idx][..., 0:1] = (coord[idx][..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
            coord[idx][..., 1:2] = (coord[idx][..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
            coord[idx][..., 2:3] = (coord[idx][..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])

            position_embedding = self.calc_emb(coord[idx])

            position_embedding = position_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)
            self.adapt_pos3d.to(position_embedding.device)
            position_embedding = self.adapt_pos3d(position_embedding)
            position_embedding = position_embedding.squeeze().squeeze()
            # 先将其进行padding
            if padding_num[idx] == 0:
                query_embedding = label_embedding.to(position_embedding.device) + position_embedding
                all_embeddings.append(query_embedding)
                continue
            else:
                if position_embedding.size()[0]==256:
                    position_embedding = position_embedding.unsqueeze(0)
                    position_embedding = torch.cat([position_embedding.to(position_embedding.device),
                                                 nn.Embedding(positive_num[idx]-1, 256).weight.to(position_embedding.device)], dim=0)
                else:
                    position_embedding = torch.cat([position_embedding,nn.Embedding(padding_num[idx], 256).weight.to(position_embedding.device)],dim=0)
                # print("labels:",labels)
                # print("label_embedding:",label_embedding.size())
                # print("padding_embedding:", nn.Embedding(padding_num[idx], 256).weight.size())
                if label_embedding.size()[0]==256:
                    label_embedding = label_embedding.unsqueeze(0)
                    label_embedding = torch.cat([label_embedding.to(position_embedding.device),nn.Embedding(positive_num[idx]-1, 256).weight.to(position_embedding.device)],dim=0)
                else:
                    label_embedding = torch.cat([label_embedding.to(position_embedding.device),nn.Embedding(padding_num[idx], 256).weight.to(position_embedding.device)], dim=0)

            query_embedding = label_embedding + position_embedding
            all_embeddings.append(query_embedding)

        # 增加 one2many query
        all_embeddings = self.one2many_query_init(point_coord, labels, pc_range, all_embeddings)

        all_embeddings = torch.stack(all_embeddings,dim=0)

        return all_embeddings