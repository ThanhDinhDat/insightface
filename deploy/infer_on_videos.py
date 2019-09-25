import face_model
import argparse
import cv2
import sys
import numpy as np
from utils import get_model, prepare_facebank, load_facebank, infer, draw_box_name, get_mtccn_faces
from config import get_config
import glob
import os
from face_detection.accuracy_evaluation import predict
from face_detection.config_farm import configuration_10_320_20L_5scales_v2 as cfg
import mxnet as mx
import numpy as np
import time

def args_parser():
    parser = argparse.ArgumentParser(description='face model test')
    # general
    parser.add_argument('--image-size', default='112,112', help='')
    parser.add_argument('--model', default='', help='path to load model.')
    parser.add_argument('--ga-model', default='', help='path to load model.')
    parser.add_argument("-s", "--save_name", help="output file name",default='recording', type=str)
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
    parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
    parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
    parser.add_argument("-f", "--file_name", help="video file path",default='', type=str)
    parser.add_argument("-b", "--begin", help="from when to start detection(in seconds)", default=0, type=int)

    conf = get_config(False)
    args = parser.parse_args()
    return args

def get_facedetector(args):
    symbol_file_path = 'face_detection/symbol_farm/symbol_10_320_20L_5scales_v2_deploy.json'
    model_file_path = 'face_detection/saved_model/configuration_10_320_20L_5scales_v2/train_10_320_20L_5scales_v2_iter_1800000.params'
    ctx = mx.gpu(args.gpu)
    face_detector = predict.Predict(mxnet=mx,
                             symbol_file_path=symbol_file_path,
                             model_file_path=model_file_path,
                             ctx=ctx,
                             receptive_field_list=cfg.param_receptive_field_list,
                             receptive_field_stride=cfg.param_receptive_field_stride,
                             bbox_small_list=cfg.param_bbox_small_list,
                             bbox_large_list=cfg.param_bbox_large_list,
                             receptive_field_center_start=cfg.param_receptive_field_center_start,
                             num_output_scales=cfg.param_num_output_scales)
    return face_detector

if __name__ == '__main__':
    args = args_parser()
    model = face_model.FaceModel(args)
    conf = get_config(training=False)

    targets, names = prepare_facebank(conf=conf, model=model, args=args, tta=True)
    if not os.path.exists(conf.demo):
        os.mkdir(conf.demo)

    cap = cv2.VideoCapture(str(args.file_name))
    
    cap.set(cv2.CAP_PROP_POS_MSEC, args.begin * 1000)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_writer = cv2.VideoWriter(str(conf.facebank_path/'{}.avi'.format(args.save_name)),
                                   cv2.VideoWriter_fourcc(*'XVID'), int(fps), (1280,720))
    # for file in glob.glob(args.folder + "/*.{}".format(args.extension)):
    while cap.isOpened():
        print('READING----------------------------')
        isSuccess,frame = cap.read()
        if not isSuccess:
            break
        else:
            result = get_mtccn_faces(args=args, ctx=mx.gpu(args.gpu), image=frame)
            if result is not None:
                aligned_faces, bboxes = result
                if len(bboxes) == 0:
                    print('no face')
                    continue
                img_size =112
                margin = 0
                img_h, img_w, _ = frame.shape

                start_time = time.time()
                results, score = infer(model=model, conf=conf, faces=aligned_faces, target_embs=targets, tta=False)
                print('Duration: {}'.format(time.time() - start_time))
                # results, score = infer(model=model, conf=conf, faces=aligned_faces, target_embs=targets, tta=True)
                
                for idx,bbox in enumerate(bboxes):
                    x1, y1, x2, y2= bbox[0], bbox[1], bbox[2] ,bbox[3]
                    xw1 = max(int(x1 - margin ), 0)
                    yw1 = max(int(y1 - margin ), 0)
                    xw2 = min(int(x2 + margin ), img_w - 1)
                    yw2 = min(int(y2 + margin ), img_h - 1)
                    bbox = [xw1, yw1, xw2,yw2]
                    # frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                    frame = draw_box_name(bbox, names[results[idx] + 1], frame)
                # frame = cv2.resize(frame, dsize=None ,fx=0.25, fy=0.25)
                
            video_writer.write(frame)
            # cv2.imshow('window', frame)
            # if cv2.waitKey(0) == ord('q'):
                # break    
    cap.release()
    video_writer.release()
