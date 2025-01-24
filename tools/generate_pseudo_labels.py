import argparse
import numpy as np
import torch
import mmcv
import os

num2cat_mapping = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
gtlabels2names = {0: 'car', 1: 'truck', 2: 'construction_vehicle', 3: 'bus', 4: 'trailer', 5: 'barrier',
                  6: 'motorcycle', 7: 'bicycle',
                  8: 'pedestrian', 9: 'traffic_cone'}

def parse_args():
    parser = argparse.ArgumentParser(description='generate pseudo label pkl')
    parser.add_argument('anno_path', help='the full train data annotation pkl path',
                        default="data/nuscenes/nuscenes_infos_val.pkl")
    parser.add_argument('pseudo_path', help='the pseudo train data predictions pkl path',
                        default="res.pkl")
    parser.add_argument('out_dir', help='the output pkl dir',
                        default="")
    parser.add_argument('--ratio', help='GT sampled rate',
                        default=0.2)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    
    ratio = float(args.ratio)
    rate_str = str(ratio).replace(".", "")
    print(f"gt ratio: {ratio}")
    
    anno_path = args.anno_path
    pseudo_path = args.pseudo_path
    out_path = os.path.join(args.out_dir, f'nuscenes_gt_pseudo_dbinfos_train_{rate_str}rate.pkl')
    
    anno_data = mmcv.load(anno_path)
    pseudo_data = mmcv.load(pseudo_path)
    print("load anno files successfully")
    
    anno_data['infos'] = list(sorted(anno_data['infos'], key=lambda e: e['timestamp']))
    
    num_samples = len(anno_data['infos'])
    print(f"num_samples: {num_samples}")
    # print(f"ratio * num_samples: {ratio * num_samples}")
    selected_num = int(ratio * num_samples)
    
    assert len(pseudo_data) == len(anno_data['infos']), (len(pseudo_data), len(anno_data['infos']))
    cnt = 0

    filter_threshold = 0
    
    for idx in range(selected_num,num_samples):
        boxes_3d = pseudo_data[idx]['pts_bbox']['boxes_3d'].tensor.numpy()
        scores_3d = pseudo_data[idx]['pts_bbox']['scores_3d'].numpy()
        labels_3d = pseudo_data[idx]['pts_bbox']['labels_3d'].numpy()
        
        boxes_3d[:,2] = boxes_3d[:,2] + boxes_3d[:,5] * 0.5
        
        if scores_3d.shape[0] >= anno_data['infos'][idx]['gt_boxes'].shape[0]:
            cnt += 1
        valid_idx = scores_3d >= filter_threshold
        anno_data['infos'][idx]['gt_boxes'] = boxes_3d[valid_idx][:,:7]
        anno_data['infos'][idx]['gt_velocity'] = boxes_3d[valid_idx][:,7:]
        # 已知label， 不需要使用pseudo label
        anno_data['infos'][idx]['gt_names'] = np.array([gtlabels2names[i] for i in labels_3d[valid_idx].astype(np.int)], dtype='<U12')
        num_selected = np.sum(valid_idx)
        dtp = anno_data['infos'][idx]['valid_flag'].dtype
        anno_data['infos'][idx]['valid_flag'] = np.array([True]* num_selected,dtype=dtp)
        anno_data['infos'][idx]['num_lidar_pts'] = np.ones((num_selected))* 100
        anno_data['infos'][idx]['num_radar_pts'] = np.ones((num_selected))* 100
        
        # print(cnt)
    mmcv.dump(anno_data, out_path)
    print(f"GT rate is {ratio}, the total samples are {num_samples}, pseudo labels are {num_samples-selected_num}")
    print(f"save anno pkl at {out_path}")
    print("generating done.")
        
if __name__ == '__main__':
    main()