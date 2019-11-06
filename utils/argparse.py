import os
import inspect
import logging

import configargparse
import torch

from utils.logging_utils import init_logger
from utils.parse_action import StoreLoggingLevelAction, CheckPathAction


class ConfigurationParer():
    """This class defines customized configuration parser
    """

    def __init__(self,
                 config_file_parser_class=configargparse.YAMLConfigFileParser,
                 formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
                 **kwargs):
        """This funtion decides config parser and formatter
        
        Keyword Arguments:
            config_file_parser_class {configargparse.ConfigFileParser} -- config file parser (default: {configargparse.YAMLConfigFileParser})
            formatter_class {configargparse.ArgumentDefaultsHelpFormatter} -- config formatter (default: {configargparse.ArgumentDefaultsHelpFormatter})
        """
        
        self.parser = configargparse.ArgumentParser(
            config_file_parser_class=config_file_parser_class,
            formatter_class=formatter_class,
            **kwargs)

    def add_save_cfgs(self):
        """This function adds saving path arguments: config file, model file...
        """

        # config file configurations
        group = self.parser.add_argument_group('Config-File')
        group.add('-config_file',
                  '--config_file',
                  required=False,
                  is_config_file_arg=True,
                  help='config file path')

        # model file configurations
        group = self.parser.add_argument_group('Model-File')
        group.add('-save_dir',
                  '--save_dir',
                  type=str,
                  required=True,
                  help='save folder.')
        group.add('-best_model_path',
                  '--best_model_path',
                  type=str,
                  action=CheckPathAction,
                  required=False,
                  help='save best model path.')
        group.add('-last_model_path',
                  '--last_model_path',
                  type=str,
                  action=CheckPathAction,
                  required=False,
                  help='load last model path.')
        group.add('-train_model_dir',
                  '--train_model_dir',
                  type=str,
                  required=True,
                  help='dir for saving temp model during training..')

    def add_data_cfgs(self):
        """This function adds dataset arguments: data file path...
        """

        self.parser.add('-dataset_name',
                        '--dataset_name',
                        type=str,
                        required=True,
                        help='dataset name.')
        self.parser.add('-dataset_dir',
                        '--dataset_dir',
                        type=str,
                        required=True,
                        help='dataset dir.')
        self.parser.add('-domain_dict',
                        '--domain_dict',
                        type=str,
                        required=True,
                        help='domain mapping dict.')
        self.parser.add('-csv_file',
                        '--csv_file',
                        type=str,
                        required=True,
                        help='origin csv data file.')
        self.parser.add('-statistic_file',
                        '--statistic_file',
                        type=str,
                        help='statistic file path.')
        self.parser.add('-event_interval_statistic_picture',
                        '--event_interval_statistic_picture',
                        type=str,
                        help='event interval statistic picture file path.')
        self.parser.add('-event_index_file',
                        '--event_index_file',
                        type=str,
                        help='event index file path.')
        self.parser.add('-min_event_interval',
                        '--min_event_interval',
                        type=float,
                        help='minimum event time interval.')
        self.parser.add('-min_length',
                        '--min_length',
                        type=int,
                        help='minimum length of event sequence.')
        self.parser.add('-max_length',
                        '--max_length',
                        type=int,
                        help='maximum length of event sequence.')
        self.parser.add('-train_rate',
                        '--train_rate',
                        type=float,
                        help='the rate of training data.')
        self.parser.add('-time_file',
                        '--time_file',
                        type=str,
                        help='time file path.')
        self.parser.add('-event_file',
                        '--event_file',
                        type=str,
                        help='event file path.')
        self.parser.add('-train_time_file',
                        '--train_time_file',
                        type=str,
                        help='training time file path.')
        self.parser.add('-train_event_file',
                        '--train_event_file',
                        type=str,
                        help='training event file path.')
        self.parser.add('-dev_time_file',
                        '--dev_time_file',
                        type=str,
                        help='validating time file path.')
        self.parser.add('-dev_event_file',
                        '--dev_event_file',
                        type=str,
                        help='validating event file path.')
        self.parser.add('-test_time_file',
                        '--test_time_file',
                        type=str,
                        help='testing time file path.')
        self.parser.add('-test_event_file',
                        '--test_event_file',
                        type=str,
                        help='testing event file path.')
        self.parser.add('-mark',
                        '--mark',
                        type=int,
                        help='indicating whether marked.')
        self.parser.add('-diff',
                        '--diff',
                        type=int,
                        help='indicating whether use event time interval.')
        self.parser.add('-save_last_time',
                        '--save_last_time',
                        type=int,
                        help='indicating whether preserve last event time.')

    def add_model_cfgs(self):
        """This function adds model (network) arguments: embedding, hidden unit...
        """

        # embedding configurations
        group = self.parser.add_argument_group('Embeddings')
        group.add('-event_classes',
                  '--event_classes',
                  type=int,
                  required=False,
                  help='event.')
        group.add('-embedding_dims',
                  '--embedding_dims',
                  type=int,
                  required=False,
                  help='embedding dimensions.')
        
        # event sequence encoder configurations
        group = self.parser.add_argument_group('Encoder')
        group.add('-lstm_hidden_unit_dims',
                  '--lstm_hidden_unit_dims',
                  type=int,
                  required=False,
                  help='lstm hidden unit dimensions.')
        group.add('-lstm_layers',
                  '--lstm_layers',
                  type=int,
                  help='lstm layers.')
        group.add('-attention_threshold',
                  '--attention_threshold',
                  type=float,
                  help='attention threshold.')
        
        # event sequence decoder configurations
        group = self.parser.add_argument_group('Decoder')
        group.add('-mlp_dims',
                  '--mlp_dims',
                  type=int,
                  required=False,
                  help='mlp dimensions')

        # regularization configurations
        group = self.parser.add_argument_group('Regularization')
        group.add('-dropout',
                  '--dropout',
                  type=float,
                  default=0.1,
                  help='dropout rate.')
    
        # loss hyperparameters
        group = self.parser.add_argument_group('Loss')
        group.add('-loss_alpha',
                  '--loss_alpha',
                  type=float,
                  default=0.05,
                  help='loss alpha.')

    def add_optimizer_cfgs(self):
        """This function adds optimizer arguements
        """

        # gradient strategy
        self.parser.add('-gradient_clipping',
                        '--gradient_clipping',
                        type=float,
                        default=1.0,
                        help='gradient clipping threshold.')

        # learning rate
        self.parser.add('--learning_rate',
                        '-learning_rate',
                        type=float,
                        default=1e-3,
                        help="Starting learning rate. "
                        "Recommended settings: sgd = 1, adagrad = 0.1, "
                        "adadelta = 1, adam = 0.001")

        # Adam configurations
        group = self.parser.add_argument_group('Adam')
        group.add('-adam_beta1',
                  '--adam_beta1',
                  type=float,
                  default=0.9,
                  help="The beta1 parameter used by Adam. "
                  "Almost without exception a value of 0.9 is used in "
                  "the literature, seemingly giving good results, "
                  "so we would discourage changing this value from "
                  "the default without due consideration.")
        group.add('-adam_beta2',
                  '--adam_beta2',
                  type=float,
                  default=0.999,
                  help='The beta2 parameter used by Adam. '
                  'Typically a value of 0.999 is recommended, as this is '
                  'the value suggested by the original paper describing '
                  'Adam, and is also the value adopted in other frameworks '
                  'such as Tensorflow and Kerras, i.e. see: '
                  'https://www.tensorflow.org/api_docs/python/tf/train/Adam'
                  'Optimizer or '
                  'https://keras.io/optimizers/ . '
                  'Whereas recently the paper "Attention is All You Need" '
                  'suggested a value of 0.98 for beta2, this parameter may '
                  'not work well for normal models / default '
                  'baselines.')
        group.add('-adam_epsilon',
                  '--adam_epsilon',
                  type=float,
                  default=1e-8,
                  help='adam epsilon')
        group.add('-adam_weight_decay_rate',
                  '--adam_weight_decay_rate',
                  type=float,
                  default=0.0,
                  help='adam weight decay rate')

    def add_run_cfgs(self):
        """This function adds running arguments
        """

        # training configurations
        group = self.parser.add_argument_group('Training')
        group.add('-seed',
                  '--seed',
                  type=int,
                  default=5216,
                  help='radom seed.')
        group.add('-epoches',
                  '--epoches',
                  type=int,
                  default=1000,
                  help='training epoches.')
        group.add('-early_stop',
                  '--early_stop',
                  type=int,
                  default=30,
                  help='early stop threshold.')
        group.add('-train_batch_size',
                  '--train_batch_size',
                  type=int,
                  default=256,
                  help='batch size during training.')
        group.add('-gradient_accumulation_steps',
                  '--gradient_accumulation_steps',
                  type=int,
                  default=1,
                  help='Number of updates steps to accumulate before performing a backward/update pass.')
        group.add('-continue_training',
                  '--continue_training',
                  action='store_true',
                  help='continue training from last.')

        # testing configurations
        group = self.parser.add_argument_group('Testing')
        group.add('-test_batch_size',
                  '--test_batch_size',
                  type=int,
                  default=100,
                  help='batch size during testing.')
        group.add('-validate_every',
                  '--validate_every',
                  type=int,
                  default=4000,
                  help='output result every n samples during validating.')

        # gpu configurations
        group = self.parser.add_argument_group('GPU')
        group.add('-device',
                  '--device',
                  type=int,
                  default=-1,
                  help='cpu: device = -1, gpu: gpu device id(device >= 0).')

        # logging configurations
        group = self.parser.add_argument_group('logging')
        group.add('-root_log_level',
                  '--root_log_level',
                  type=str,
                  action=StoreLoggingLevelAction,
                  choices=StoreLoggingLevelAction.CHOICES,
                  default="DEBUG",
                  help='root logging out level.')
        group.add('-console_log_level',
                  '--console_log_level',
                  type=str,
                  action=StoreLoggingLevelAction,
                  choices=StoreLoggingLevelAction.CHOICES,
                  default="NOTSET",
                  help='console logging output level.')
        group.add('-log_file',
                  '--log_file',
                  type=str,
                  action=CheckPathAction,
                  required=True,
                  help='logging file during running.')
        group.add('-file_log_level',
                  '--file_log_level',
                  type=str,
                  action=StoreLoggingLevelAction,
                  choices=StoreLoggingLevelAction.CHOICES,
                  default="NOTSET",
                  help='file logging output level.')
        group.add('-logging_steps',
                  '--logging_steps',
                  type=int,
                  default=10,
                  help='Logging every N update steps.')

    def parse_args(self):
        """This function parses arguments and initializes logger
        
        Returns:
            dict -- config arguments
        """

        cfg = self.parser.parse_args()
        init_logger(root_log_level=getattr(cfg, 'root_log_level',
                                           logging.DEBUG),
                    console_log_level=getattr(cfg, 'console_log_level',
                                              logging.NOTSET),
                    log_file=getattr(cfg, 'log_file', None),
                    log_file_level=getattr(cfg, 'log_file_level',
                                           logging.NOTSET))
        
        if not os.path.exists(cfg.save_dir):
            os.makedirs(cfg.save_dir)

        if not os.path.exists(cfg.train_model_dir):
            os.makedirs(cfg.train_model_dir)

        return cfg

    def format_values(self):
        return self.parser.format_values()
