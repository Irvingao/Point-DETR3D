import os
import mmcv
import argparse
import torch

def parse_args():
    parser = argparse.ArgumentParser(description='fuse two checkpoints')
    parser.add_argument('--data_root', default='data/nuscenes', help='the sampled train dbinfo pkl path')
    parser.add_argument('--train_data_pkl', default='nuscenes_infos_train.pkl', help='the train data pkl path')
    parser.add_argument('--dbinfos_train_pkl', default='nuscenes_dbinfos_train.pkl', help='the train dbinfo pkl path')
    parser.add_argument('--split_ratio', default=0.25, type=float, help='the sampled train dbinfo pkl path')

    args = parser.parse_args()

    return args


class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

def main():
    args = parse_args()
    data_root = args.data_root
    split_ratio = args.split_ratio
    train_data_pkl = os.path.join(data_root, args.train_data_pkl)
    dbinfos_train_pkl = os.path.join(data_root, args.dbinfos_train_pkl)
    out_pkl = os.path.join(data_root, f'nuscenes_sampled_{str(split_ratio).replace(".", "")}ratio_dbinfos_train.pkl')
    
    data = mmcv.load(train_data_pkl)
    print(f"Read train data from {train_data_pkl}, {len(data['infos'])} data in total.")
    dbinfo = mmcv.load(dbinfos_train_pkl)
    print(f"Read db info from {dbinfos_train_pkl}")

    end_idx = int(len(data['infos']) // (1/split_ratio))
    
    all_token = []

    for idx in range(end_idx):
        all_token.append(data['infos'][idx]['token'])

    sampled_dbinfo = { 'car':[], 'truck':[], 'construction_vehicle':[], 'bus':[], 'trailer':[], 'barrier':[],
        'motorcycle':[], 'bicycle':[], 'pedestrian':[], 'traffic_cone':[]}

    for key in dbinfo.keys():
        if key in class_names:

            for info_id in range(len(dbinfo[key])):
                if dbinfo[key][info_id]['image_idx'] in all_token:
                    sampled_dbinfo[key].append(dbinfo[key][info_id])

    mmcv.dump(sampled_dbinfo, out_pkl)
    print(f"Sampling done, saved at {out_pkl}.")
    
main()