import os
import sys
import argparse
import input_data
import tensorflow as tf

# os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '5'

FLAGS = None

ps_hosts = ["localhost:3333"]
worker_hosts = ["localhost:3334", "localhost:3335"]

def main(_):
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    server = tf.train.Server(cluster, job_name   = FLAGS.job_name,
                                      task_index = FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()
    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
            worker_device = "/job:worker/task:%d" % FLAGS.task_index,
            cluster = cluster)):

            # build module #
            mnist = input_data.read_data_sets("../data/mnist_data/", one_hot=True)
            
            d           = 784
            m           = 10
            x           = tf.placeholder("float", shape = [None, d], name = 'input')
            ref         = tf.placeholder("float", shape = [None, m], name = 'label')
            weight      = tf.Variable(tf.zeros([d, m]),              name = 'weight')
            bias        = tf.Variable(tf.zeros([m]),                 name = 'bias') 
            y           = tf.nn.softmax(tf.matmul(x, weight) + bias)
            global_step = tf.contrib.framework.get_or_create_global_step()
            
            loss        = -tf.reduce_sum(ref * tf.log(y))
            train_step  = tf.train.GradientDescentOptimizer(0.01).minimize(
                              loss, global_step = global_step)
            correct     = tf.equal(tf.argmax(y, 1), tf.argmax(ref, 1))
            accuracy    = tf.reduce_mean(tf.cast(correct, "float"))
            init_op     = tf.global_variables_initializer()

        # handle stopping after given steps #
        last_step = 100
        hooks = [tf.train.StopAtStepHook(last_step = last_step)]

        # MonitoredTrainingSession take care of session #
        with tf.train.MonitoredTrainingSession(master = server.target,
                                               is_chief = (FLAGS.task_index == 0), # main task, init, checkpoint, save, restore
                                               hooks = hooks) as monitor_sess:
            while not monitor_sess.should_stop():
                batch = mnist.train.next_batch(100)
                _, step = monitor_sess.run([train_step, global_step],
                                           feed_dict = {x: batch[0], ref: batch[1]})
                if step != last_step:
                    acc = monitor_sess.run(accuracy, feed_dict = {x: mnist.test.images, ref: mnist.test.labels})
                    print("Step %d in task %d with accuracy %f" % (step, FLAGS.task_index, acc))
                    # print("Step %d in task %d" % (step, FLAGS.task_index))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_name",     type = str, default = "")
    parser.add_argument("--task_index",   type = int, default = 0)

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main = main, argv = [sys.argv[0]] + unparsed)

