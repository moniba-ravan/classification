from .main import main_args


def mobilenet_args():
    parser = main_args()
    return parser.parse_args()
