import argparse
import glob

import tensorflow.compat.v1 as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from  object_detection.protos import preprocessor_pb2


def edit(train_dir, eval_dir, batch_size, checkpoint, label_map):
    """
    edit the config file and save it to pipeline_new.config
    args:
    - train_dir [str]: path to train directory
    - eval_dir [str]: path to val OR test directory 
    - batch_size [int]: batch size
    - checkpoint [str]: path to pretrained model
    - label_map [str]: path to labelmap file
    """
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig() 
    with tf.gfile.GFile("pipeline.config", "r") as f:                                                                                                                                                                                                                     
        proto_str = f.read()                                                                                                                                                                                                                                          
        text_format.Merge(proto_str, pipeline_config)  
    
    training_files = glob.glob(train_dir + '/*.tfrecord')
    evaluation_files = glob.glob(eval_dir + '/*.tfrecord')

    pipeline_config.train_config.batch_size = batch_size
    pipeline_config.train_config.fine_tune_checkpoint = checkpoint
    pipeline_config.train_input_reader.label_map_path = label_map
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = training_files

    pipeline_config.eval_input_reader[0].label_map_path = label_map
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = evaluation_files

    ##### Augmentation #####

    pipeline_config.train_config.data_augmentation_options.normalize_image.original_minval=0
    pipeline_config.train_config.data_augmentation_options.normalize_image.original_maxval=255
    pipeline_config.train_config.data_augmentation_options.normalize_image.target_minval=0
    pipeline_config.train_config.data_augmentation_options.normalize_image.target_maxval=1

    pipeline_config.train_config.data_augmentation_options.random_horizontal_flip.probability=0.5

    pipeline_config.train_config.data_augmentation_options.random_adjust_brightness.max_delta=0.2

    pipeline_config.train_config.data_augmentation_options.random_adjust_contrast.min_delta=0.8
    pipeline_config.train_config.data_augmentation_options.random_adjust_contrast.max_delta=1.25

    pipeline_config.train_config.data_augmentation_options.random_adjust_hue.max_delta=0.02

    pipeline_config.train_config.data_augmentation_options.random_adjust_saturation.min_delta=0.8
    pipeline_config.train_config.data_augmentation_options.random_adjust_saturation.max_delta=1.25

    ########################    

    config_text = text_format.MessageToString(pipeline_config)             
    with tf.gfile.Open("pipeline_new_aug.config", "wb") as f:                                                                                                                                                                                                                       
        f.write(config_text)   


if __name__ == "__main__": 

    if True:        
        parser = argparse.ArgumentParser(description='Download and process tf files')
        parser.add_argument('--train_dir', required=True, type=str,
                            help='training directory')
        parser.add_argument('--eval_dir', required=True, type=str,
                            help='validation or testing directory')
        parser.add_argument('--batch_size', required=True, type=int,
                            help='number of images in batch')
        parser.add_argument('--checkpoint', required=True, type=str,
                            help='checkpoint path')   
        parser.add_argument('--label_map', required=True, type=str,
                            help='label map path')   
        args = parser.parse_args()
        edit(args.train_dir, args.eval_dir, args.batch_size, 
             args.checkpoint, args.label_map)
        