# train_junk_net 

## arguments

-i /jasper/data/train_data_sets/junk_detection/tfrecord/ -o /jasper/result/training/junk_detection/checkpoints/ -g 8

## tensorboard

tensorboard --logdir /jasper/result/training/junk_detection/checkpoints/

# eval_junk_net 

## arguments

-i /jasper/data/train_data_sets/junk_detection/tfrecord/ -c /jasper/result/training/junk_detection/checkpoints/ -o /jasper/result/training/junk_detection/eval_dir/

## tensorboard

tensorboard --logdir /jasper/result/training/junk_detection/checkpoints/ --port 6007
