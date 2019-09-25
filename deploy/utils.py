import mxnet as mx
import numpy as np
import time
from mtcnn_detector import MtcnnDetector
import face_preprocess
import os
import cv2
import torch
from PIL import Image
import mtcnn
from torchvision import transforms as trans


def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix   = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model


def prepare_facebank(conf, model, args, tta = True):
    # ctx = mx.gpu(args.gpu)
    # det_threshold = [0.6,0.7,0.8]
    # mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    # detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=det_threshold)
    
    
    #--------------------MODEL------------------------
    embeddings =  []
    names = ['Unknown']
    for path in conf.facebank_path.iterdir():
        if path.is_file():
            continue
        else:
            embs = []
            for filename in path.iterdir():
                if not filename.is_file():
                    continue
                else:
                    try:
                        print(filename)
                        pil_image = Image.open(filename)
                        np_img = np.array(pil_image)
                        print('---------------------------------------------')                        
                    except Exception  as e:
                        print(e)
                        continue
                    
                    
                    with torch.no_grad():
                        if tta:
                            np_img = model.get_input(np_img)
                            feature = model.get_feature(np_img)

                            horizontal_flip = np.flip(np_img,0)
                            horizontal_np_img = model.get_input(horizontal_flip)
                            
                            if horizontal_np_img is not None:
                                horizontal_feature = model.get_feature(horizontal_np_img)
                                feature = l2_norm(feature + horizontal_feature)
                            embs.append(feature)
                            embeddings.append(feature)
                            
                        else:
                            np_img = model.get_input(np_img)
                            feature = model.get_feature(np_img)
                            # print('Original feature: {}'.format(feature))
                            torch_feature = torch.FloatTensor([feature])          
                            embs.append(feature)
                            embeddings.append(feature)
                            # embs.append(torch_feature)
        if len(embs) == 0:
            continue
        # embedding = torch.cat(embs).mean(0,keepdim=True)
        # embedding = np.concatenate(embs).mean(0, keepdims=True)
        # print('After concat: {}'.format(embedding))
        # embeddings.append(embedding)
        # embeddings.append(embs)
        names.append(path.name)
    # embeddings = np.concatenate(embeddings)
    names = np.array(names)
    # torch.save(embeddings, conf.facebank_path/'facebank.pth')
    np.save(conf.facebank_path/'facebank', embeddings)
    np.save(conf.facebank_path/'names', names)
    return embeddings, names

def load_facebank(conf):
    embeddings = torch.load(conf.facebank_path/'facebank.npy')
    names = np.load(conf.facebank_path/'names.npy')
    return embeddings, names

def draw_box_name(bbox,name,frame):
    if 'Unknown' != name.split('_')[0]:
        frame = cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[2],bbox[3]),(0,0,255),6)
        frame = cv2.putText(frame,
                        name,
                        (bbox[0],bbox[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        2,
                        (0,255,0),
                        3,
                        cv2.LINE_AA)
    return frame

def get_mtccn_faces(args, ctx, image):
    # det_threshold = [0.6,0.7,0.8]
    det_threshold = [0.0,0.1,0.2]
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    aligned_faces = []
    detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=det_threshold)
    print('Input shape: {}'.format(image.shape))
    ret = detector.detect_face(image, det_type = args.det)
    if ret is None:
      return None
    bboxes, points = ret

    if bboxes.shape[0]==0:
      return None
    for index, bbox in enumerate(bboxes):
        point = points[index]
        point = point.reshape((2,5)).T
        
        
        nimg = face_preprocess.preprocess(image, bbox, point, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        # cv2.imshow('window', nimg)
        # cv2.waitKey(0)
        aligned = np.transpose(nimg, (2,0,1))
        aligned_faces.append(aligned)
    return aligned_faces, bboxes

def infer(model, conf, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in facebank
        names : recorded names of faces in facebank
        tta : test time augmentation (hfilp, that's all)
        '''
        embs = []
        source_embs = []
        for img in faces:
            if tta:
                print('Type: {}'.format(type(img)))
                feature = model.get_feature(img)
                
                horizontal_flip = np.flip(img,0)
                horizontal_np_img = model.get_input(horizontal_flip)
                embs.append(l2_norm(emb + emb_mirror))
            else:
                start_time = time.time()
                feature = model.get_feature(img)     
                print('Feature extraction duration: {}'.format(time.time() - start_time))   
                embs.append(feature)
                source_embs.append(feature)
        similar = []
        # print(source_embs)
        for source_array in source_embs:
            sim = []
            source_transpose = source_array.T
            for target_array in target_embs:
                sim.append(np.dot(source_transpose, target_array))
            similar.append(sim)
        sims = np.array(similar)
        # print(sims)
        maxi = [np.amax(sim) for sim in sims]
        maximum = np.array(maxi)
        min_idx = [np.where(sim == np.amax(sim))[0][0] for sim in sims]
        for index in np.where(maximum < 0.0)[0]:
            min_idx[index] = -1
        print('Similarity dist: {}'.format(min_idx))
        return min_idx, maxi               