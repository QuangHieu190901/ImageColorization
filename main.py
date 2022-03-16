import cv2
import numpy as np
from skimage import data

# https://github.com/richzhang/colorization/blob/caffe/colorization/models/colorization_deploy_v2.prototxt
# http://eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v2/colorization_release_v2.caffemodel
# https://github.com/richzhang/colorization/blob/caffe/colorization/resources/pts_in_hull.npy
# https://github.com/opencv/opencv/blob/master/samples/dnn/colorization.py

def colorized_img_process(grayscale_red_filename, grayscale_green_filename, grayscale_blue_filename):
    r = cv2.imread(grayscale_red_filename, 0)
    g = cv2.imread(grayscale_green_filename, 0)
    b = cv2.imread(grayscale_blue_filename, 0)

    resized = (512, 512)
    r = cv2.resize(r, resized)
    g = cv2.resize(g, resized)
    b = cv2.resize(b, resized)

    colorized = cv2.merge((r, g, b))
    colorized = cv2.cvtColor(colorized, cv2.COLOR_BGR2RGB)

    cv2.imshow('img', colorized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def colorize_ml(img):

    # paths
    prototxt_path = 'model/colorization_deploy_v2.prototxt'
    model_path = 'model/colorization_release_v2.caffemodel'
    kernel_path = 'model/pts_in_hull.npy'

    # load model
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    points = np.load(kernel_path)
    points = points.transpose().reshape(2, 313, 1, 1)
    net.getLayer(net.getLayerId('class8_ab')).blobs = [points.astype('float32')]
    net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, dtype= 'float32')]    

    # normalized img, convert to LAB color space, resize to 224x224 and reduce brightness 
    normalized = img.astype('float32')/255
    lab = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # feed img to model
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0)) 
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    L = cv2.split(lab)[0]

    # return colorized image
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis= 2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_Lab2BGR)
    colorized = (255 * colorized).astype('uint8')

    # compare 2 images
    # result = np.concatenate((img, colorized), axis= 1)

    cv2.imshow('img', colorized)
    # cv2.imshow('img', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()       

if __name__ == '__main__':
    # img = cv2.imread('rocket.jpg')
    # colorize_ml(img)
    colorized_img_process('r.jpg', 'g.jpg', 'b.jpg')