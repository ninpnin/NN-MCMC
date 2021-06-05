import tensorflow as tf
import tensorflow_probability as tfp

x = tf.constant([1., -3.4, -2.])
y = tf.constant([-3., -5.3])

@tf.function
def target_log_prob_fn(W1, W2, b1, b2):
  x1 = tf.tensordot(W1, x, axes=[0,0]) + b1
  y1 = tf.nn.relu(x1)

  x2 = tf.tensordot(W2, y1, axes=[0,0]) + b2

  prior  = tf.reduce_sum(tf.multiply(W1, W1))
  prior += tf.reduce_sum(tf.multiply(W2, W2))
  prior += tf.reduce_sum(tf.multiply(b1, b1))
  prior += tf.reduce_sum(tf.multiply(b2, b2))

  prior = prior * 1.
  diff = x2 - y
  return tf.reduce_sum(tf.multiply(diff, diff))

A1 = tf.random.normal([3, 2])
A2 = tf.random.normal([2, 2])
C1 = tf.random.normal([2])
C2 = tf.random.normal([2])

y_prime = target_log_prob_fn(A1, A2, C1, C2)
print(y_prime)

print(tf.size(y_prime))

num_results = 5000
num_burnin_steps = 2000
@tf.function(autograph=False, experimental_compile=True)
def do_sampling():
  return tfp.mcmc.sample_chain(
      num_results=num_results,
      num_burnin_steps=num_burnin_steps,
      current_state=[
          tf.random.normal([3, 2], name='W1'),
          tf.random.normal([2, 2], name='W2'),
          tf.random.normal([2], name='b1'),
          tf.random.normal([2], name='b2'),
      ],
      kernel=tfp.mcmc.HamiltonianMonteCarlo(
          target_log_prob_fn=target_log_prob_fn,
          step_size=0.4,
          num_leapfrog_steps=3))


states, kernel_results = do_sampling()

print(len(states))
print(type(states))

print(states[0])
print(type(states[0]))

print(states[1])
print(type(states[1]))

print(type(kernel_results))
print(len(kernel_results))