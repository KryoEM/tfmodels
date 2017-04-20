import tensorflow as tf
import math

import  junk_net
import  arg_parsing
from    datasets import dataset_utils
from    junk_preprocessing import get_preprocess_image

slim = tf.contrib.slim
args = arg_parsing.parse_args(training=False)

tf.logging.set_verbosity(tf.logging.INFO)

with tf.Graph().as_default() as g:
    with g.device(args.eval_device):
        split_name  = 'test'
        dataset     = dataset_utils.get_dataset(split_name, args.data_dir)
        params      = dataset_utils.read_meta_params(split_name, args.data_dir)
        num_classes = dataset.num_classes

        tf_global_step = slim.get_or_create_global_step()

        provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            shuffle=False,
            common_queue_capacity=2 * args.batch_size,
            common_queue_min=args.batch_size)

        [image, label] = provider.get(['image', 'label'])

        image_preprocessing_fn = get_preprocess_image(is_training=False)

        image = image_preprocessing_fn(image, params['height'], params['width'])

        images, labels = tf.train.batch(
            [image, label],
            batch_size=args.batch_size,
            num_threads=args.reader_threads,
            capacity=5 * args.batch_size)

        logits, end_points = junk_net.inference(images,num_classes,is_training=False)

        predictions = tf.argmax(logits, 1)

        # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'accuracy': slim.metrics.streaming_accuracy(predictions, labels),
            'precision': slim.metrics.streaming_precision(predictions, labels),
            'recall': slim.metrics.streaming_recall(predictions, labels),
        })
        # Print the summaries to screen.
        for name, value in names_to_values.iteritems():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        summary_op = tf.summary.merge_all()

        # accuracy, update_op     = slim.metrics.streaming_accuracy(predictions,labels)
        # recall, update_op       = slim.metrics.streaming_recall(predictions,labels)
        # precision, update_op    = slim.metrics.streaming_precision(predictions,labels)
        #
        # op = tf.summary.scalar('eval/accuracy',accuracy,collections=[])
        # op = tf.Print(op, [accuracy], 'eval/accuracy/print')
        # tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        # tf.summary.scalar('eval/recall',recall)
        # tf.summary.scalar('eval/precision',precision)
        # summary_op = tf.summary.merge_all()
        #
        num_batches = math.ceil(dataset.num_samples / float(args.batch_size))

        sess_config = tf.ConfigProto(allow_soft_placement=True)
        slim.evaluation.evaluation_loop(
        #slim.evaluation.evaluate_once(
            master='',
            checkpoint_dir=args.checkpoint_dir,
            logdir=args.output_eval_dir,
            num_evals=num_batches,
            eval_op=names_to_updates.values(),
            eval_interval_secs=args.eval_interval_secs,
            session_config=sess_config,
            variables_to_restore=slim.get_variables_to_restore())
