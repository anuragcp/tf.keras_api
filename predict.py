import tensorflow as tf
import os
import pickle
from scipy import misc
import numpy as np

classifier_f = open("int_to_word_out.pickle", "rb")
int_to_word_out = pickle.load(classifier_f)
classifier_f.close()

def pre_process(image):
    image = image.astype('float32')
    image = image / 255.0
    return image


def load_image():

    img=os.listdir("predict")[0]
    image=np.array(misc.imread("predict/"+img))
    image = misc.imresize(image, (64, 64))
    image=np.array([image])
    image=pre_process(image)
    #return image
    print("loaded image")


# Loading model and checkpoints
with tf.Session() as sess:
    saver = tf.train.import_meta_graph("./TF_Model/tf_model.meta")
    saver.restore(sess=sess, save_path="./TF_Model/tf_model")
    graph = tf.get_default_graph()
    input_fn = graph.get_tensor_by_name(name="input_fn_input:0")
    softmax_layer = graph.get_tensor_by_name(name="softmax/Softmax:0")
    print("all is well")
    feed_img = load_image()
    print(sess.run(softmax_layer,feed_dict=feed_img))