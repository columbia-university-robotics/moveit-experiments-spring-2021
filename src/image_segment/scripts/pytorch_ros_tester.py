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
from sensor_msgs.msg import PointCloud2, Image, PointField
import sensor_msgs.msg as sensor_msgs # used in point_cloud(...)
import std_msgs.msg as std_msgs # used in point_cloud(...)
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
        rospy.logerr("temp_xyzrgb_array "+str(temp_xyzrgb_array.dtype.names))                    
        rospy.logerr("temp_xyzrgb_array "+str(temp_xyzrgb_array.dtype))                       
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


def point_cloud(points, parent_frame="camera_rgb_optical_frame"):
    """ Creates a point cloud message.
    https://gist.github.com/pgorczak/5c717baa44479fa064eb8d33ea4587e0
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = sensor_msgs.PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    fields = [sensor_msgs.PointField(
        name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
        for i, n in enumerate('xyzrgba')]

    header = std_msgs.Header(frame_id=parent_frame, stamp=rospy.Time.now())

    return sensor_msgs.PointCloud2(
        header=header,
        height=480,
        width=640,
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=(itemsize * 7),
        row_step=(itemsize * 7 * points.shape[0]),
        data=data
    )    
    
def enough_time_has_passed():
    global LAST_IMAGE_UPDATE, CURRENTLY_UPDATING
    return not CURRENTLY_UPDATING and rospy.get_time() - LAST_IMAGE_UPDATE > SAMPLE_INTERVAL 
    
     
def main():
    global LAST_IMAGE_UPDATE, CURRENTLY_UPDATING, RGB_IMAGE, DEPTH_IMAGE, HAS_RUN_THROUGH_SEGMENTER
    uois_net_3d = get_network()
    pub_segment = rospy.Publisher("/image_segment/uois/seg_masks", Image, queue_size=2) 
    pub_centers = rospy.Publisher("/image_segment/uois/center_offsets", PointCloud2, queue_size=2) 
       
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
            fg_masks  =  fg_masks.cpu().numpy()
            center_offsets = center_offsets.cpu().numpy().transpose(0,2,3,1)
            initial_masks  =  initial_masks.cpu().numpy()
            
            # publish
            num_objs = np.unique(seg_masks[0]).max() + 1    
            seg_mask_plot = util_.get_color_mask(seg_masks[0], nc=num_objs)    
            
            #rospy.logerr(str(fg_masks[0].shape)+str(center_offsets[0].shape)+str(initial_masks[0].shape)+str(seg_masks[0].shape)+"\nseg_mask_plot"+str(seg_mask_plot.shape))
            #rospy.logerr("fg "+str(fg_masks.dtype)+"\ncenter "+str(center_offsets.dtype)+"\ninit "+str(initial_masks.dtype)+"\nseg "+str(seg_masks.dtype)+"\nseg_mask_plot"+str(seg_mask_plot.dtype))            
            #rospy.logerr("DEPTH_IMAGE "+str(DEPTH_IMAGE.dtype))      
            
            #rospy.logerr("center_offs"+str(center_offsets[0].reshape((-1,3))))                  
            msg_ar = np.array([center_offsets[0][:,:,0],center_offsets[0][:,:,1],center_offsets[0][:,:,2],seg_mask_plot[:,:,0],seg_mask_plot[:,:,1],seg_mask_plot[:,:,2],np.ones(seg_mask_plot[:,:,2].shape)],dtype=np.float32)        
            # https://github.com/eric-wieser/ros_numpy/blob/master/test/test_images.py
            data = seg_mask_plot
            msg  = ros_numpy.msgify(Image, data,encoding='rgb8')
            pub_segment.publish(msg)
            
            msg  = point_cloud(msg_ar)
            pub_centers.publish(msg)  
                      
            # making sure not to overload the system
            total_time = time() - st_time 
            rospy.logdebug("total_time : "+str(round(total_time, 3)))# 480 , 640
            HAS_RUN_THROUGH_SEGMENTER = True
            
        r.sleep() 

if __name__ == "__main__":
    rospy.init_node('pytorch_ros_tester_node', anonymous=True)
    
    rospy.Subscriber("/camera/depth_registered/points", PointCloud2, pc_image_cb)
    main()    
    rospy.spin()                                
    
