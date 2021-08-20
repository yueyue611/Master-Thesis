import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Sequential


# You can also create random constant tensors
x = tf.random.normal(shape=(2, 2), mean=0.0, stddev=1.0)
print(x)
x = tf.random.uniform(shape=(2, 2), minval=0, maxval=10, dtype="int32")
print(x)

initial_value = tf.random.normal(shape=(2, 2))
a = tf.Variable(initial_value)
print("a: ", a)

new_value = tf.random.normal(shape=(2, 2))
a.assign(new_value)
print("new value and new a: ", new_value, a)

added_value = tf.random.normal(shape=(2, 2))
a.assign_add(added_value)
print("added value and new a: ", added_value, a)


b = tf.random.normal(shape=(2, 2))
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as tape:
        c = tf.sqrt(tf.square(a) + tf.square(b))
        dc_da = tape.gradient(c, a)
    d2c_da2 = outer_tape.gradient(dc_da, a)
    print(d2c_da2)


class Linear(layers.Layer):
    """y = w.x + b"""

    def __init__(self, units=32, input_dim=32):
        super(Linear, self).__init__()
        # w_init = tf.random_normal_initializer()
        w_init = tf.ones_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.ones_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


# Instantiate our layer.
linear_layer = Linear(units=4, input_dim=3)

# The layer can be treated as a function.
# Here we call it on some data.
y = linear_layer(tf.ones((2, 3)))  # shape(x, input_dim)
print("y: ", y,
      "\nw: ", linear_layer.w,  # shape(input_dim, units)
      "\nb: ", linear_layer.b)   # shape(units,) batches of units-dimensional vector


# Here's a layer with a non-trainable weight:
class ComputeSum(layers.Layer):
    """Returns the sum of the inputs."""

    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        # Create a non-trainable weight.
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))  # the tensor is reduced along the first dimension (rows)
        return self.total


my_sum = ComputeSum(2)
x = tf.ones((2, 2))

y = my_sum(x)
print(y.numpy())  # [2. 2.]
print(my_sum.weights)

z = my_sum(x)
print(z.numpy())  # [4. 4.]

# Prepare our layer, loss, and optimizer.
model = Sequential(
    [
        layers.Dense(32, activation="relu"),
        layers.Dense(32, activation="relu"),
        layers.Dense(10),
    ]
)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)


# Create a training step function.
@tf.function  # Make it fast.
def train_on_batch(x, y):
    with tf.GradientTape() as tape:
        logits = model(x)
        loss = loss_fn(y, logits)
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    return loss


# Prepare a dataset.
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
dataset = tf.data.Dataset.from_tensor_slices(
    (x_train.reshape(60000, 784).astype("float32") / 255, y_train)
)
dataset = dataset.shuffle(buffer_size=1024).batch(64)

for step, (x, y) in enumerate(dataset):
    loss = train_on_batch(x, y)
    if step % 100 == 0:
        print("Step:", step, "Loss:", float(loss))
