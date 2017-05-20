__author__ = 'Yonatan Dishon'

"""
This is default configuration file for segnet_generic model
The user is supposed to supply user_config.py in the model directory
in order to run the model with his specific path and visualization configurations
The user_config.py will not be commited to git (is ignored in .gitignore of tfmodels)
"""

import importlib

###################################################################################
# Mandatory parameters:

# Example of default configuration and explaination of each field:
# Must supply paths:
mandatory_params = ['rawdata_dir',
                    'tfrecord_dir',
                    'checkpoint_dir',
                    'log_dir',
                    'output_dir']
rawdata_dir    =  'Path of raw images to create tfrecords'
tfrecord_dir   =  'Path of dataset (path to tfrecords directory)'
checkpoint_dir =  'Path to save .ckpt model files and log files (training)'
log_dir        =  'Path to save log files (evaluation)'
output_dir    =  'directory to save evaluation output results (if applicable)'


mandatory_dict = dict( (name,eval(name)) for name in mandatory_params )

####################################################################################
# Model dependent parameters:

# Train Parameters
user_params_train = ['num_gpus',
                     'print_log_steps',
                     'save_checkpoint_secs',
                     'save_summaries_secs',
                     'reader_threads']

# Evaluation Parameters
user_params_eval = ['batch_size',
                    'eval_interval_secs',
                    'reader_threads',
                    'use_gpu']

# tfrecorder parameters
user_params_recorder = ['train_split',
                        'test_split',
                        'num_shards',
                        'print_step_tfrec']
######################################################


def getconfig(config_nm, flags_dict):
    """
    The get user configuration interface function - checks if user_config.py is available and check
    if all model parameters (from above are set) if not it will override them
    :param config_nm: configuration name
    :param flags_dict: the entire '__flags' dictionary given by the command line/train_generic/eval_generic etc
    :return:
    dictionary with configuration fields
    """

    try:
        user_config = importlib.import_module('.user_config', '.'.join(str(__name__).split('.')[:-1]))
        final_config = override_config(config_nm, user_config, flags_dict)
        return final_config

    except ImportError:
        print('There isn\'t a user_config.py in your model folder!\nPlease supply one based on default_config.py')
        _print_config()


def override_config(config_nm, usr_conf, flags_dict):
    """
    Validate that mandatory params are in usr_conf and overriding user_params from usr_conf over the default_config
    :param usr_conf: a module specifing configuration
    :param flags_dict: the entire '__flags' dictionary given by the command line/train_generic/eval_generic etc
    :return: usr_conf with override params
    """
    # check all mandatory params are here:
    for p, v in mandatory_dict.iteritems():
        if not hasattr(usr_conf, p):
            print('User should supply path for %s: %s' % (p, v))
            exit(-1)
    user_params = getdefflags(config_nm)
    # check if optional user defined params are listed
    # if not just use the specified params
    for p, v in flags_dict.iteritems():
        if p in user_params:
            if not hasattr(usr_conf,  p):
                setattr(usr_conf, p, v)

    return usr_conf


def getdefflags(config_nm):
    """
    return a list of default parameters -  user and mandatory
    :param config_nm: the configuration of which to bring
    :return:
    """
    if config_nm is 'train':
        user_params = user_params_train
    elif config_nm is 'eval':
        user_params = user_params_eval
    elif config_nm is 'tfrecorder':
        user_params = user_params_recorder
    else:
        print('Unrecognized configuration name : %s, exiting ....' % config_nm)
        exit(-1)


    return mandatory_params+user_params


# Helper functions for printing configuration to the stdout
def _get_book_variable_module_name():
    module = globals()
    book = {}
    if module:
        book = {key: value for key, value in module.iteritems() if not (key.startswith('__') or key.startswith('_') or str(value).startswith('<'))}
    return book


def _print_config():
    book = _get_book_variable_module_name()
    for key, value in book.iteritems():
        print "{:<30}{:<100}".format(key, value)

if __name__ == '__main__':
    getconfig()

