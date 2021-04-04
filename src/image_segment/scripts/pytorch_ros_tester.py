#!/usr/bin/env python3

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "0" # TODO: Change this if you have more than 1 GPU

import sys
import json
from time import time
import glob

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

# My libraries. Ugly hack to import from sister directory
import image_segment.data_augmentation as data_augmentation
import image_segment.segmentation as segmentation
import image_segment.evaluation as evaluation
import image_segment.utilities as util_
import image_segment.flowlib as flowlib
# ROS imports
import rospy
from sensor_msgs.msg import PointCloud2, Image 
import ros_numpy

CURRENTLY_UPDATING = False
SAMPLE_INTERVAL = 20
LAST_IMAGE_UPDATE = 0
RGB_IMAGE = None
DEPTH_IMAGE = None
HAS_RUN_THROUGH_SEGMENTER = True # init to true
def get_model_configs():

    dsn_config = {
        
        # Sizes
        'feature_dim' : 64, # 32 would be normal

        # Mean Shift parameters (for 3D voting)
        'max_GMS_iters' : 10, 
        'epsilon' : 0.05, # Connected Components parameter
        'sigma' : 0.02, # Gaussian bandwidth parameter
        'num_seeds' : 200, # Used for MeanShift, but not BlurringMeanShift
        'subsample_factor' : 5,
        
        # Misc
        'min_pixels_thresh' : 500,
        'tau' : 15.,
        
    }

    rrn_config = {
        
        # Sizes
        'feature_dim' : 64, # 32 would be normal
        'img_H' : 224,
        'img_W' : 224,
        
        # architecture parameters
        'use_coordconv' : False,
        
    }

    uois3d_config = {
        
        # Padding for RGB Refinement Network
        'padding_percentage' : 0.25,
        
        # Open/Close Morphology for IMP (Initial Mask Processing) module
        'use_open_close_morphology' : True,
        'open_close_morphology_ksize' : 9,
        
        # Largest Connected Component for IMP module
        'use_largest_connected_component' : True,
        
    }

    return dsn_config, rrn_config, uois3d_config

def get_network():
    dsn_config, rrn_config, uois3d_config = get_model_configs()
    #rospy.logerr(os.path.abspath('.'))
    #TODO CHANGE SO THAT EVERYONE CAN LOAD WITH RELATIVE PATH
    absolute_path = '/home/biobe/workspace/moveit-experiments-spring-2021/src/image_segment/'
    checkpoint_dir = absolute_path+'checkpoints/'#'/home/chrisxie/projects/uois/checkpoints/' # TODO: change this to directory of downloaded models
    dsn_filename = checkpoint_dir + 'DepthSeedingNetwork_3D_TOD_checkpoint.pt'
    rrn_filename = checkpoint_dir + 'RRN_OID_checkpoint.pt'
    uois3d_config['final_close_morphology'] = 'TableTop_v5' in rrn_filename
    uois_net_3d = segmentation.UOISNet3D(uois3d_config, 
                                         dsn_filename,
                                         dsn_config,
                                         rrn_filename,
                                         rrn_config
                                        )
    return uois_net_3d


def old_main():
    uois_net_3d = get_network()
    """                                    
    example_images_dir = absolute_path + 'example_images/'

    OSD_image_files = sorted(glob.glob(example_images_dir + '/OSD_*.npy'))
    OCID_image_files = sorted(glob.glob(example_images_dir + '/OCID_*.npy'))
    N = len(OSD_image_files) + len(OCID_image_files)

    rgb_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)
    xyz_imgs = np.zeros((N, 480, 640, 3), dtype=np.float32)
    label_imgs = np.zeros((N, 480, 640), dtype=np.uint8)

    for i, img_file in enumerate(OSD_image_files + OCID_image_files):
        d = np.load(img_file, allow_pickle=True, encoding='bytes').item()
        rospy.logerr(str(d.keys()))
        # RGB
        rgb_img = d['rgb']
        rgb_imgs[i] = data_augmentation.standardize_image(rgb_img)

        # XYZ
        xyz_imgs[i] = d['xyz']

        # Label
        label_imgs[i] = d['label']
        
    batch = {
        'rgb' : data_augmentation.array_to_tensor(rgb_imgs),
        'xyz' : data_augmentation.array_to_tensor(xyz_imgs),
    }


    print("Number of images: {0}".format(N))

    ### Compute segmentation masks ###
    st_time = time()
    fg_masks, center_offsets, initial_masks, seg_masks = uois_net_3d.run_on_batch(batch)
    total_time = time() - st_time
    print('Total time taken for Segmentation: {0} seconds'.format(round(total_time, 3)))
    print('FPS: {0}'.format(round(N / total_time,3)))

    # Get results in numpy

    seg_masks = seg_masks.cpu().numpy()
    fg_masks = fg_masks.cpu().numpy()
    center_offsets = center_offsets.cpu().numpy().transpose(0,2,3,1)
    initial_masks = initial_masks.cpu().numpy()
    

    rgb_imgs = util_.torch_to_numpy(batch['rgb'].cpu(), is_standardized_image=True)
    total_subplots = 6
    """
    
    
    
    
    """
    fig_index = 1
    for i in range(N):
        
        num_objs = max(np.unique(seg_masks[i,...]).max(), np.unique(label_imgs[i,...]).max()) + 1
        
        rgb = rgb_imgs[i].astype(np.uint8)
        depth = xyz_imgs[i,...,2]
        seg_mask_plot = util_.get_color_mask(seg_masks[i,...], nc=num_objs)
        gt_masks = util_.get_color_mask(label_imgs[i,...], nc=num_objs)
        
        images = [rgb, depth, seg_mask_plot, gt_masks]
        titles = [f'Image {i+1}', 'Depth',
                  f"Refined Masks. #objects: {np.unique(seg_masks[i,...]).shape[0]-1}",
                  f"Ground Truth. #objects: {np.unique(label_imgs[i,...]).shape[0]-1}"
                 ]
        util_.subplotter(images, titles, fig_num=i+1)
        
        # Run evaluation metric
        eval_metrics = evaluation.multilabel_metrics(seg_masks[i,...], label_imgs[i])
        print(f"Image {i+1} Metrics:")
        print(eval_metrics)
    """    
    rospy.logerr("Finished with pytorch_ros_tester_node")
        
def pc_image_cb(im):
    """
    rostopic pub /camera/depth_registered/points sensor_msgs/PointCloud2 "header:
      seq: 0
      stamp: {secs: 0, nsecs: 0}
      frame_id: ''
    height: 0
    width: 0
    fields:
    - {name: '', offset: 0, datatype: 0, count: 0}
    is_bigendian: false
    point_step: 0
    row_step: 0
    data: !!binary ""
    is_dense: false"
    """
    global LAST_IMAGE_UPDATE, CURRENTLY_UPDATING, RGB_IMAGE, DEPTH_IMAGE, HAS_RUN_THROUGH_SEGMENTER
    
    if enough_time_has_passed() and HAS_RUN_THROUGH_SEGMENTER:

        CURRENTLY_UPDATING = True
        LAST_IMAGE_UPDATE = rospy.get_time()
        
        
        # https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/point_cloud2.py
        # get numpy array from msg
        pc_void_array = ros_numpy.point_cloud2.pointcloud2_to_array(im)
        temp_xyzrgb_array = ros_numpy.point_cloud2.split_rgb_field(pc_void_array)
        xyz_array = ros_numpy.point_cloud2.get_xyz_points(temp_xyzrgb_array,remove_nans=False)

        # reshape so that the ndtype is not made of void types
        h,w = temp_xyzrgb_array['r'].shape 
        assert((h,w) == (480,640))
        assert(xyz_array.shape == (h,w,3))
        rgb_array = np.zeros((h,w,3))
        rgb_array[:,:,0] = temp_xyzrgb_array['r']
        rgb_array[:,:,1] = temp_xyzrgb_array['g']
        rgb_array[:,:,2] = temp_xyzrgb_array['b']   
         
        # update the latest image
        RGB_IMAGE = rgb_array
        DEPTH_IMAGE = xyz_array
        CURRENTLY_UPDATING = False
        HAS_RUN_THROUGH_SEGMENTER = False

        
def rgb_image_cb(im):
    #rostopic pub /camera/color/image_raw sensor_msgs/Image "header:
    #  seq: 0
    #  stamp: {secs: 0, nsecs: 0}
    #  frame_id: ''
    #height: 0
    #width: 0
    #encoding: ''
    #is_bigendian: 0
    #step: 0
    #data: !!binary 
    
    #rospy.logerr("rgb_image_cb : "+str(im.height)+str(im.width))# 480 , 640
    numpy_rgb_image = ros_numpy.numpify(im)
    
def depth_image_cb(im):
    #rostopic pub /camera/depth/image_raw sensor_msgs/Image "header:
    #  seq: 0
    #  stamp: {secs: 0, nsecs: 0}
    #  frame_id: ''
    #height: 0
    #width: 0
    #encoding: ''
    #is_bigendian: 0
    #step: 0
    #data: !!binary 
    #rospy.logerr("depth_image_cb : "+str(im.height)+str(im.width))# 480 , 640
    numpy_depth_image = ros_numpy.numpify(im)
    
    
    
def enough_time_has_passed():
    global LAST_IMAGE_UPDATE, CURRENTLY_UPDATING
    return not CURRENTLY_UPDATING and rospy.get_time() - LAST_IMAGE_UPDATE > SAMPLE_INTERVAL 
    
     
def main():
    global LAST_IMAGE_UPDATE, CURRENTLY_UPDATING, RGB_IMAGE, DEPTH_IMAGE, HAS_RUN_THROUGH_SEGMENTER
    uois_net_3d = get_network()
    
    rgb_imgs = np.zeros((1, 480, 640, 3), dtype=np.float32)
    xyz_imgs = np.zeros((1, 480, 640, 3), dtype=np.float32)    
    
    r = rospy.Rate(10) # 10hz
    
    while not rospy.is_shutdown():
        if enough_time_has_passed() :
            st_time = time()
            
            rgb_imgs[0] = RGB_IMAGE
            xyz_imgs[0] = DEPTH_IMAGE
  
            batch = {
                'rgb' : data_augmentation.array_to_tensor(rgb_imgs),
                'xyz' : data_augmentation.array_to_tensor(xyz_imgs),
            }

            ### Compute segmentation masks ###
            fg_masks, center_offsets, initial_masks, seg_masks = uois_net_3d.run_on_batch(batch)

            # Get results in numpy
            seg_masks = seg_masks.cpu().numpy()
            fg_masks = fg_masks.cpu().numpy()
            center_offsets = center_offsets.cpu().numpy().transpose(0,2,3,1)
            initial_masks = initial_masks.cpu().numpy()
            rospy.logerr(str(fg_masks.shape)+str(center_offsets.shape)+str(initial_masks.shape)+str(seg_masks.shape))
            total_time = time() - st_time 
            rospy.logdebug("total_time : "+str(round(total_time, 3)))# 480 , 640
            HAS_RUN_THROUGH_SEGMENTER = True
            
        r.sleep() 

if __name__ == "__main__":
    rospy.init_node('pytorch_ros_tester_node', anonymous=True)
    
    rospy.Subscriber("/camera/depth_registered/points", PointCloud2, pc_image_cb)
    rospy.Subscriber("/camera/depth/image_raw", Image, depth_image_cb)
    rospy.Subscriber("/camera/color/image_raw", Image, rgb_image_cb)

    main()    
    rospy.spin()                                
    
