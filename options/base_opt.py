# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/options/base_options.py

import argparse
import os
from utils.general import mkdirs

class BaseOptions():
    """This class defines options used during both training and test time.
    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--lr', type=float, default=1e-3,help='learning-rate for training')
        parser.add_argument('--batch-size', type=int, default=8,help='batch size for training')
        parser.add_argument('--num-epochs', type=int, default=1000,help='number of epoch for training')
        parser.add_argument('--num-workers', type=int, default=8)
        parser.add_argument('--seed', type=int, default=123)
        parser.add_argument('--factor', type=int, default=2,help='downsample factor for training')
        parser.add_argument('--dataset_size', default='full', type=str,
                    help='datset size',choices=['resolution25','resolution50'])             

        # additional parameters
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--tensorboard', action='store_true', help='if specified, log information in tensorboard')
       
        self.initialized = True
        return parser

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, 'command_line_opt'+str(opt.factor))
        mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # save and return the parser
        self.parser = parser
        return parser.parse_args() 

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        self.print_options(opt)
        self.opt = opt
        return self.opt