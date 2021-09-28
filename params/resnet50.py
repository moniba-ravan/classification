from .main import main_args


def resnet50_args():
    parser = main_args()
    return parser.parse_args()
