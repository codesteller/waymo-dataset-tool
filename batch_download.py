import os
import argparse
import multiprocessing
import numpy as np
import tools.waymo as wod
import 


def main():

    parser = argparse.ArgumentParser(description='Download and convert Waymo dataset to KITTI format')
    parser.add_argument('--out_dir', type=str, default='./waymo', help='Output directory')
    parser.add_argument('--split', type=str, default='training', help='training or validation')
    parser.add_argument('--resize', type=float, default=1, help='Resize ratio')
    args = parser.parse_args()

    if args.split == 'training':
        num_segs = 32
    elif args.split == 'validation':
        num_segs = 8
    else:
        raise ValueError('Invalid split: %s'%args.split)

    os.makedirs(args.out_dir, exist_ok=True)

    labels_dir = os.path.join(args.out_dir, 'labels')
    images_dir = os.path.join(args.out_dir, 'images')
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    seg_ids = np.random.choice(100, args.num_seg, replace=False)
    seg_ids.sort()
    for seg_id in seg_ids:
        
        wod.download_and_convert(args, seg_id, labels_dir, images_dir)


    print("Done")



if __name__ == '__main__':
    main()