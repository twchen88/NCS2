import tensorflow as tf
print(tf.__version__)

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

loaded = tf.saved_model.load('handwritten_model')
infer = loaded.signatures['serving_default']

f = tf.function(infer).get_concrete_function(flatten_input=tf.TensorSpec(shape=[None, 784], dtype=tf.float32))
f2 = convert_variables_to_constants_v2(f)
graph_def = f2.graph.as_graph_def()

# Export frozen graph
with tf.io.gfile.GFile('frozen_graph.pb', 'wb') as f:
   f.write(graph_def.SerializeToString())
