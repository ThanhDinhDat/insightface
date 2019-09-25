from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

def get_config(training = True):
    conf = edict()
    conf.data_path = Path('../datasets/data')
    conf.work_path = Path('work_space/')
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log'
    conf.save_path = conf.work_path/'save'
    conf.input_size = [112,112]
    conf.embedding_size = 512
    # conf.use_mobilfacenet = True
    conf.use_mobilfacenet = False
    # conf.net_depth = 50
    conf.net_depth = 100
    conf.drop_ratio = 0.6
    # conf.net_mode = 'ir_se' # or 'ir'
    conf.net_mode = 'ir_se' # or 'ir'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([
                    trans.ToTensor(),
                    trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])
    conf.data_mode = 'emore'
    conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path/'faces_emore'
    conf.batch_size = 100 # irse net depth 50 
#--------------------Inference Config ------------------------
    conf.facebank_path = conf.data_path/'facebank_copy_1'
    conf.threshold = 1
    conf.face_limit = 10 
    #when inference, at maximum detect 10 faces in one image, my laptop is slow
    conf.min_face_size = 50 
    # the larger this value, the faster deduction, comes with tradeoff in small faces
    conf.demo = Path('./demo/')
    return conf