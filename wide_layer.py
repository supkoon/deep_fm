import tensorflow as tf
from tensorflow import keras
class wide_part(keras.layers.Layer):
    def __init__(self, V, num_fields, embedding_lookup_index, **kwargs):
        super().__init__(self, **kwargs)
        self.V = V
        self.num_fields = num_fields
        self.embedding_lookup_index = embedding_lookup_index

    def build(self, batch_input_size):
        w_init = tf.random_normal_initializer()

        self.W = tf.Variable(initial_value=w_init(shape=[batch_input_size[-1]]),
                             dtype='float32',name = "W")
        self.V = tf.Variable(initial_value=w_init(shape=[self.num_fields, self.V]),
                             dtype="float32",name= "V")

    def call(self, inputs):
        # (None,108)
        x_batch = keras.layers.Reshape((inputs.shape[-1], 1))(inputs)
        # (None,108,1)
        embeddings_lookup_table = tf.nn.embedding_lookup(params=self.V, ids=self.embedding_lookup_index)
        # (108,V) --> embedding_lookup_table

        embedded_fields = tf.math.multiply(x_batch, embeddings_lookup_table)
        # element-wise after broadcasting to (None,108,1) --> (None,108,V)

        order_1_output = tf.reduce_sum(tf.math.multiply(inputs, self.W), axis=1)
        #         elementwise after broadcasting (None,108) x (108) = None,108
        #         reduce_sum == (None,)

        embed_sum = tf.reduce_sum(embedded_fields, [1, 2])
        # (None,108,V) == > (None,)
        embed_square = tf.square(embedded_fields)
        # (None,108,V) ==> (None,108,V)
        square_of_sum = tf.square(embed_sum)
        # (None,) == > (None,)
        sum_of_square = tf.reduce_sum(embed_square, [1, 2])
        # (None,108,V) == > (None, )
        order_2_output = 0.5 * tf.subtract(square_of_sum, sum_of_square)
        # (None,) ==> (None,)
        order_1_output = keras.layers.Reshape((-1, 1))(order_1_output)
        # (None,) ==> (None,1,1)
        order_2_output = keras.layers.Reshape((-1, 1))(order_2_output)
        # (None,) ==> (None,1,1)
        wide_output = keras.layers.Concatenate(axis=1)([order_1_output, order_2_output])
        #         print(order_1_output.shape)
        #         print(order_2_output.shape)
        # (None,2,1)

        linear_terms = tf.reduce_sum(
            tf.math.multiply(self.W, inputs), axis=1, keepdims=False)

        # (batch_size, )
        interactions = 0.5 * tf.subtract(
            tf.square(tf.reduce_sum(embedded_fields, [1, 2])),
            tf.reduce_sum(tf.square(embedded_fields), [1, 2])
        )

        linear_terms = tf.reshape(linear_terms, [-1, 1])
        interactions = tf.reshape(interactions, [-1, 1])

        wide_output = tf.concat([linear_terms, interactions], 1)

        return wide_output, embedded_fields

