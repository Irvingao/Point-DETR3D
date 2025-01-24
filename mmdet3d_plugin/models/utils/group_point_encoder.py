
import copy
import math

import mmcv
import torch
import torch.nn as nn


class GroupPointEncoder(nn.Module):

    def __init__(self,num_classes,num_feats,point_coord,labels,num_group,scale=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_feats = num_feats
        self.point_coord = point_coord
        self.labels = labels
        self.num_group = num_group
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

    def add_noise(self,pc_range):
        gtlabels2names = {0: 'car', 1: 'truck', 2: 'construction_vehicle', 3: 'bus', 4: 'trailer', 5: 'barrier',
                          6: 'motorcycle', 7: 'bicycle',
                          8: 'pedestrian', 9: 'traffic_cone'}
        group_point_coord = []
        group_labels = []
        small_category = [0,6,7,8,9]
        large_category = [1,2,3,4,5]
        
        # Hard code for eval 
        if isinstance(self.labels, list):
            labels = self.labels 
        else:
            labels = [self.labels]
        
        for group_id in range(self.num_group):
            point_coord = copy.deepcopy(self.point_coord)
            all_noise_point_coord = []
            for batch_id in range(len(self.point_coord)):
                for idx in range(len(self.point_coord[batch_id])):
                    if labels[batch_id][idx] in small_category:
                        noise = torch.normal(0,2,size=(1,3),device=point_coord[batch_id].device)
                    elif labels[batch_id][idx] in large_category:
                        noise = torch.normal(0,4, size=(1,3),device=point_coord[batch_id].device)
                    else:
                        noise = 0

                    point_coord[batch_id][idx] = point_coord[batch_id][idx]+noise

                all_noise_point_coord.append(copy.deepcopy(point_coord[batch_id]))
            group_point_coord.append(all_noise_point_coord)
            group_labels.append(labels)
        group_point_coord[0] = self.point_coord

        return group_point_coord,group_labels

    def forward(self, point_coord,labels,pc_range):

        batch_size = len(point_coord)
        positive_num = [point_coord[idx].size(0) for idx in range(batch_size)]
        padding_num = [max(positive_num) - positive_num[idx] for idx in range(batch_size)]

        group_point_coord, group_labels = self.add_noise(pc_range)
        # mmcv.dump(group_point_coord,'group_point_coord.pkl')
        for group_id in range(self.num_group):

            for batch_id in range(len(self.point_coord)):
                group_point_coord[group_id][batch_id][..., 0:1] = (group_point_coord[group_id][batch_id][..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
                group_point_coord[group_id][batch_id][..., 1:2] = (group_point_coord[group_id][batch_id][..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
                group_point_coord[group_id][batch_id][..., 2:3] = (group_point_coord[group_id][batch_id][..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])
        # mmcv.dump(group_point_coord,'data.pkl')
        group_embeding = []
        for group_id in range(self.num_group):
            all_embeddings = []
            for batch_id in range(batch_size):
                label_embedding = self.label_embed.weight[group_labels[group_id][batch_id].long()]

                position_embedding = self.calc_emb(group_point_coord[group_id][batch_id])
                position_embedding = position_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)
                self.adapt_pos3d.to(position_embedding.device)
                position_embedding = self.adapt_pos3d(position_embedding)
                position_embedding = position_embedding.squeeze().squeeze()
                if padding_num[batch_id]==0:
                    query_embedding = label_embedding.to(position_embedding.device) + position_embedding
                    query_pos = nn.Embedding(max(positive_num), self.num_feats * 2)
                    query_embedding = torch.cat((query_pos.weight.to(query_embedding.device), query_embedding), dim=1)
                    all_embeddings.append(query_embedding)
                else:
                    if position_embedding.size()[0] == 256:
                        position_embedding = position_embedding.unsqueeze(0)
                        position_embedding = torch.cat([position_embedding,
                                                        nn.Embedding(padding_num[batch_id], 256).weight.to(
                                                            position_embedding.device)], dim=0)
                    else:
                        position_embedding = torch.cat([position_embedding,
                                                        nn.Embedding(padding_num[batch_id], 256).weight.to(
                                                            position_embedding.device)], dim=0)
                    if label_embedding.size()[0] == 256:
                        label_embedding = label_embedding.unsqueeze(0)

                        label_embedding = torch.cat([label_embedding.to(position_embedding.device),
                                                     nn.Embedding(padding_num[batch_id], 256).weight.to(
                                                         position_embedding.device)], dim=0)
                    else:
                        label_embedding = torch.cat([label_embedding.to(position_embedding.device),
                                                     nn.Embedding(padding_num[batch_id], 256).weight.to(
                                                         position_embedding.device)], dim=0)

                    query_embedding = label_embedding.to(position_embedding.device) + position_embedding

                        # 再cancat一个[num_point,256]
                        # num_points = query_embedding.shape[0]
                        # query_pos = nn.Embedding(300, self.num_feats*2)
                    query_pos = nn.Embedding(max(positive_num), self.num_feats * 2)
                    query_embedding = torch.cat((query_pos.weight.to(query_embedding.device), query_embedding),dim=1)
                    all_embeddings.append(query_embedding)

            group_embeding.append(torch.stack(all_embeddings,dim=0))

        group_embeding = torch.cat(group_embeding,dim=0)
        return group_embeding,group_point_coord,group_labels


class H_GroupPointEncoder(nn.Module):

    def __init__(self,num_classes,num_feats,point_coord,labels,num_group,k_one2many,scale=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_feats = num_feats
        self.point_coord = point_coord
        self.labels = labels
        self.num_group = num_group
        self.k_one2many = k_one2many
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


    def add_noisev2(self, pc_range):
        gtlabels2names = {0: 'car', 1: 'truck', 2: 'construction_vehicle', 3: 'bus', 4: 'trailer', 5: 'barrier',
                          6: 'motorcycle', 7: 'bicycle',
                          8: 'pedestrian', 9: 'traffic_cone'}
        group_point_coord = []
        group_labels = []
        small_category = [0, 6, 7, 8, 9]
        large_category = [1, 2, 3, 4, 5]
        bs = len(self.point_coord)
        positive_num = [self.point_coord[idx].size()[0] for idx in range(bs)]
        padding_num = [max(positive_num) - positive_num[idx] for idx in range(bs)]

        for group_id in range(self.num_group):
            point_coord = copy.deepcopy(self.point_coord)
            all_noise_point_coord = []
            for batch_id in range(len(self.point_coord)):

                for idx in range(len(self.point_coord[batch_id])):
                    if self.labels[batch_id].shape == torch.Size([]):
                        noise = torch.tensor(0, device=self.point_coord[batch_id].device)
                    else:
                        if self.labels[batch_id][idx] in small_category:
                            noise = torch.normal(0, 2, size=(1, 3), device=point_coord[batch_id].device)
                        if self.labels[batch_id][idx] in large_category:
                            noise = torch.normal(0, 4, size=(1, 3), device=point_coord[batch_id].device)
                    point_coord[batch_id][idx] = point_coord[batch_id][idx] + noise

                all_noise_point_coord.append(copy.deepcopy(point_coord[batch_id]))
            group_point_coord.append(all_noise_point_coord)
            group_labels.append(self.labels)
        group_point_coord[0] = self.point_coord

        new_group_point_coord = []
        new_group_labels = []
        for group_id in range(self.num_group):
            new_point_coord = []
            new_labels = []
            for batch_id in range(bs):
                if padding_num[batch_id]==0:
                    new_point_coord.append(group_point_coord[group_id][batch_id])
                    if group_labels[group_id][batch_id].shape ==torch.Size([]):
                        new_labels.append(group_labels[group_id])
                    else:
                        new_labels.append(group_labels[group_id][batch_id])
                    continue
                else:
                    new_point_coord.append(torch.cat((group_point_coord[group_id][batch_id],torch.zeros(padding_num[batch_id],3,device=group_point_coord[group_id][batch_id].device)),
                                           dim=0))
                    pad_label = torch.randint(low=0, high=10, size=(padding_num[batch_id], 1)).squeeze(-1)
                    pad_label = pad_label.to(group_labels[group_id][batch_id].device)
                    new_labels.append(torch.cat((group_labels[group_id][batch_id],pad_label), dim=0))

            new_group_point_coord.append(new_point_coord)
            new_group_labels.append(new_labels)


        one2many_num = self.k_one2many*max(positive_num)
        # 直接随机生成one2many 的 referencepoints
        for group_id in range(self.num_group):
            for batch_id in range(bs):

                # one2many_ref = torch.randn(size=(one2many_num,3),device=new_group_point_coord[group_id][batch_id].device)
                # one2many_ref[..., 0:1] = one2many_ref[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
                # one2many_ref[..., 1:2] = one2many_ref[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
                #
                # tmp_coord = torch.cat((new_group_point_coord[group_id][batch_id],one2many_ref))
                # new_group_point_coord[group_id][batch_id] = tmp_coord
                new_group_point_coord[group_id][batch_id] = new_group_point_coord[group_id][batch_id].repeat(self.k_one2many+1, 1)

                # one2many_label =torch.randint(low=0, high=10, size=(one2many_num, 1)).squeeze(-1)
                # one2many_label = one2many_label.to(new_group_labels[group_id][batch_id].device)
                # if new_group_labels[group_id][batch_id].shape == torch.Size([]):
                #     tmp_label = torch.randint(low=0, high=10, size=(300, 1)).squeeze(-1).to(new_group_labels[group_id][batch_id].device)
                # else:
                #     tmp_label = torch.cat((new_group_labels[group_id][batch_id], one2many_label))

                new_group_labels[group_id][batch_id] = new_group_labels[group_id][batch_id].repeat(self.k_one2many+1)

        return new_group_point_coord,new_group_labels

    def forward(self, point_coord,labels,pc_range):

        batch_size = len(point_coord)
        positive_num = [point_coord[idx].size(0) for idx in range(batch_size)]
        padding_num = [max(positive_num) - positive_num[idx] for idx in range(batch_size)]
        one2many_num = self.k_one2many*max(positive_num)
        group_point_coord, group_labels = self.add_noisev2(pc_range)
        # mmcv.dump(group_point_coord,'group_point_coord.pkl')
        for group_id in range(self.num_group):

            for batch_id in range(len(self.point_coord)):
                group_point_coord[group_id][batch_id][..., 0:1] = (group_point_coord[group_id][batch_id][..., 0:1] - pc_range[0]) / (pc_range[3] - pc_range[0])
                group_point_coord[group_id][batch_id][..., 1:2] = (group_point_coord[group_id][batch_id][..., 1:2] - pc_range[1]) / (pc_range[4] - pc_range[1])
                group_point_coord[group_id][batch_id][..., 2:3] = (group_point_coord[group_id][batch_id][..., 2:3] - pc_range[2]) / (pc_range[5] - pc_range[2])
        # mmcv.dump(group_point_coord,'data.pkl')
        group_embeding = []
        for group_id in range(self.num_group):
            all_embeddings = []
            for batch_id in range(batch_size):
                label_embedding = self.label_embed.weight[group_labels[group_id][batch_id].long()]

                position_embedding = self.calc_emb(group_point_coord[group_id][batch_id])
                position_embedding = position_embedding.unsqueeze(dim=-1).unsqueeze(dim=-1)
                self.adapt_pos3d.to(position_embedding.device)
                position_embedding = self.adapt_pos3d(position_embedding)
                position_embedding = position_embedding.squeeze().squeeze()

                if position_embedding.size()[0] == 256:
                    position_embedding = position_embedding.unsqueeze(0)


                if label_embedding.size()[0] == 256:
                    label_embedding = label_embedding.unsqueeze(0)

                query_embedding = label_embedding.to(position_embedding.device) + position_embedding

                # 再cancat一个[num_point,256]
                # num_points = query_embedding.shape[0]
                # query_pos = nn.Embedding(300, self.num_feats*2)
                query_pos = nn.Embedding(max(positive_num)*(self.k_one2many+1), self.num_feats * 2)
                query_embedding = torch.cat((query_pos.weight.to(query_embedding.device), query_embedding),dim=1)
                all_embeddings.append(query_embedding)

            group_embeding.append(torch.stack(all_embeddings,dim=0))
        # for group_id in range(self.num_group):
        #     group_embeding[group_id] = self.one2many_query_init(self.point_coord, labels, pc_range, group_embeding[group_id])
        #     # group_embeding[group_id] = torch.stack(group_embeding[group_id],dim=0)
        group_embeding = torch.cat(group_embeding,dim=0)
        return group_embeding,group_point_coord,group_labels