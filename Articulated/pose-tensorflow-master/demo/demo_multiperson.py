import os
import sys

import numpy as np

sys.path.append(os.path.dirname(__file__) + "/../")

from scipy.misc import imread, imsave
from config import load_config
from dataset.factory import create as create_dataset
from nnet import predict
from util import visualize
from dataset.pose_dataset import data_to_input

from multiperson.detections import extract_detections
from multiperson.predict import SpatialModel, eval_graph, get_person_conf_multicut
from multiperson.visualize import PersonDraw, visualize_detections


import matplotlib
import matplotlib.pyplot as plt


cfg = load_config("demo/pose_cfg_multi.yaml")

dataset = create_dataset(cfg)

sm = SpatialModel(cfg)
sm.load()

draw_multi = PersonDraw()

# Load and setup CNN part detector
sess, inputs, outputs = predict.setup_pose_prediction(cfg)
list_name=os.listdir('/home/cgv841/otakudj/MyWork/MOT/Data/2DMOT2015/train/')
for i in range(len(list_name)):
    path='/home/cgv841/otakudj/MyWork/MOT/Data/2DMOT2015/train/'+list_name[i]+'/img1'
    if not os.path.exists('img/'+list_name[i]):
        os.makedirs('img/'+list_name[i])
    for j in range(len(os.listdir(path))):
        # Read image from file
        file_name = path+"/%06d"%(j+1)+".jpg"
        image = imread(file_name, mode='RGB')
        image_batch = data_to_input(image)

        # Compute prediction with the CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref, pairwise_diff = predict.extract_cnn_output(outputs_np, cfg, dataset.pairwise_stats)

        detections = extract_detections(cfg, scmap, locref, pairwise_diff)
        unLab, pos_array, unary_array, pwidx_array, pw_array = eval_graph(sm, detections)
        person_conf_multi = get_person_conf_multicut(sm, unLab, unary_array, pos_array)

        img = np.copy(image)

        visim_multi = img.copy()

        fig = plt.imshow(visim_multi)

        draw_multi.draw(visim_multi, dataset, person_conf_multi)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)


        plt.savefig('img/'+list_name[i]+'/%06d'%(j+1)+'.jpg')
        #plt.close()
        plt.show()
        #visualize.waitforbuttonpress()


'''
        if not os.path.exists(list_name[i]):
            os.makedirs(list_name[i]+'/Articulated')
        np.savetxt(list_name[i]+'/Articulated/'+'%06d'%(j+1)+'_x.txt',person_conf_multi[:,:,0],fmt='%d',delimiter=',')
        np.savetxt(list_name[i]+'/Articulated/'+'%06d'%(j+1)+'_y.txt',person_conf_multi[:,:,1],fmt='%d',delimiter=',')
        '''
print('@')


'''
img = np.copy(image)

visim_multi = img.copy()

fig = plt.imshow(visim_multi)

draw_multi.draw(visim_multi, dataset, person_conf_multi)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)

plt.show()

visualize.waitforbuttonpress()
'''



