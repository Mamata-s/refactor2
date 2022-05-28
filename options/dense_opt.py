from .base_opt import BaseOptions


class DenseOptions(BaseOptions):
    """This class includes test options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--weights-file', type=str,help='weights file to resume the training')
        parser.add_argument('--growth-rate', type=int, default=7)
        parser.add_argument('--num-blocks', type=int, default=5)
        parser.add_argument('--num-layers', type=int, default=5)
        parser.add_argument('--scale', type=int, default=4)

        # parser.add_argument('--patch-size', type=int, default=100)
        parser.add_argument('--lr', type=float, default=1e-3)

        return parser