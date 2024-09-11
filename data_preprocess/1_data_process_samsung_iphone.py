import os
import scipy.io
import argparse
from util import split_large_image,space_to_depth,depth_to_space
from PIL import Image
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type',choices=['zoom','split'],default='split')
    parser.add_argument('--processed_image_size',type=list,default=[512,384])
    parser.add_argument('--patch_size',default=256) # 四通道rggb图片size
    parser.add_argument('--parent_path',default='/data2/xiedongyu/raw_to_raw/')
    parser.add_argument('--paired_data_path',default='dataset/paired')
    parser.add_argument('--unpaired_data_path',default='dataset/unpaired')
    parser.add_argument('--processed_data_path',default='dataset/processed')
    parser.add_argument('--camera1',default='iphone-x')
    parser.add_argument('--camera2',default='samsung-s9')
    parser.add_argument('--anchor_path',default='anchor-raw-rggb')
    parser.add_argument('--raw_path',default='raw-rggb')
    args = parser.parse_args()
    args.paired_data_path = args.parent_path + args.paired_data_path
    args.unpaired_data_path = args.parent_path + args.unpaired_data_path
    args.processed_data_path = args.parent_path + args.processed_data_path

    if not os.path.exists(args.processed_data_path):
        os.mkdir(args.processed_data_path)
        os.mkdir(os.path.join(args.processed_data_path,'train'))
        os.mkdir(os.path.join(args.processed_data_path,'train',args.camera1))
        os.mkdir(os.path.join(args.processed_data_path,'train',args.camera2))
        os.mkdir(os.path.join(args.processed_data_path,'train','vis-'+args.camera1))
        os.mkdir(os.path.join(args.processed_data_path,'train','vis-'+args.camera2))

        os.mkdir(os.path.join(args.processed_data_path,'test'))
        os.mkdir(os.path.join(args.processed_data_path,'test',args.camera1))
        os.mkdir(os.path.join(args.processed_data_path,'test',args.camera2))
        os.mkdir(os.path.join(args.processed_data_path,'test','vis-'+args.camera1))
        os.mkdir(os.path.join(args.processed_data_path,'test','vis-'+args.camera2))

        os.mkdir(os.path.join(args.processed_data_path,'unpaired'))
        os.mkdir(os.path.join(args.processed_data_path,'unpaired',args.camera1))
        os.mkdir(os.path.join(args.processed_data_path,'unpaired',args.camera2))
        os.mkdir(os.path.join(args.processed_data_path,'unpaired','vis-'+args.camera1))
        os.mkdir(os.path.join(args.processed_data_path,'unpaired','vis-'+args.camera2))
    index = 0
    train_raws = os.listdir(os.path.join(args.paired_data_path, args.camera1, 'anchor-raw-rggb'))
    test_raws = os.listdir(os.path.join(args.paired_data_path, args.camera1, 'raw-rggb'))
    dataset_type = 'train'
    for raws in [train_raws,test_raws]:
        for raw in raws:
            s_index = index
            for camera in [args.camera1,args.camera2]:
                camera_raw = scipy.io.loadmat(os.path.join(args.paired_data_path, camera, args.anchor_path if dataset_type=='train'
                                                           else args.raw_path, raw))['raw_rggb']
                camera_raw_splits = split_large_image(image=camera_raw, patch_size=args.patch_size,type=args.type,processed_image_size=args.processed_image_size)
                index = s_index
                for camera_raw_split in camera_raw_splits:
                    camera_raw_split = camera_raw_split.transpose(2,0,1)
                    np.save(os.path.join(args.processed_data_path,dataset_type,camera,str(index)+'.npy'),camera_raw_split)
                    camera_raw_split = depth_to_space(camera_raw_split).squeeze(0)
                    image = Image.fromarray((camera_raw_split * 255).astype(np.uint8))
                    image.save(os.path.join(args.processed_data_path,dataset_type,'vis-'+camera,str(index)+'.png'))
                    index += 1
        dataset_type = 'test'
        index = 0

    index = 0
    unpaired_raws_c1 = os.listdir(os.path.join(args.unpaired_data_path, args.camera1, 'raw-rggb'))
    unpaired_raws_c2 = os.listdir(os.path.join(args.unpaired_data_path, args.camera2, 'raw-rggb'))
    camera = args.camera1
    dataset_type = 'unpaired'
    for raws in [unpaired_raws_c1,unpaired_raws_c2]:
        for raw in raws:
            camera_raw = scipy.io.loadmat(os.path.join(args.unpaired_data_path, camera, args.raw_path, raw))['raw_rggb']
            camera_raw_splits = split_large_image(image=camera_raw, patch_size=args.patch_size,type=args.type,processed_image_size=args.processed_image_size)
            for camera_raw_split in camera_raw_splits:
                camera_raw_split = camera_raw_split.transpose(2, 0, 1)
                np.save(os.path.join(args.processed_data_path, dataset_type, camera, str(index) + '.npy'),
                        camera_raw_split)
                camera_raw_split = depth_to_space(camera_raw_split).squeeze(0)
                image = Image.fromarray((camera_raw_split * 255).astype(np.uint8))
                image.save(os.path.join(args.processed_data_path, dataset_type, 'vis-' + camera, str(index) + '.png'))
                index += 1
        index = 0
        camera = args.camera2
