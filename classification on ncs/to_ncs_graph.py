import tensorflow.keras as keras
import tensorflow.keras.backend as K
import tensorflow as tf

model_file = "model_face.json"
weights_file = "model_face.h5"
config=None

with open(model_file, "r") as file:
    config = file.read()

K.set_learning_phase(0)
model = keras.models.model_from_json(config)
model.load_weights(weights_file)

saver = tf.train.Saver()
sess = K.get_session()
saver.save(sess, "./TF_Model/tf_model")

fw = tf.summary.FileWriter('logs', sess.graph)
fw.close()
