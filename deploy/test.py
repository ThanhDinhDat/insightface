import face_model
import argparse
import cv2
import sys
import numpy as np

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
parser.add_argument('--img1', default='Tom_Hanks_54745.png', help='path to load model.')
parser.add_argument('--img2', default='', help='path to load model.')

args = parser.parse_args()

model = face_model.FaceModel(args)
img = cv2.imread(args.img1)
img = model.get_input(img)
f1 = model.get_feature(img)
print(f1[0:10])
# gender, age = model.get_ga(img)
# print(gender)
# print(age)
# sys.exit(0)
img = cv2.imread(args.img2)
print('-----------------')
img = model.get_input(img)
print('Image input size: {}'.format(img.shape))
    
f2 = model.get_feature(img)
print(f2[0:10])

dist = np.sum(np.square(f1-f2))
print('Euclidian: {}'.format(dist))
sim = np.dot(f1, f2.T)
print('Similarity: {}'.format(sim))
#diff = np.subtract(source_feature, target_feature)
#dist = np.sum(np.square(diff),1)
