from argparse import ArgumentParser

def resnet50_args():
    parser = ArgumentParser()
    
    parser.add_argument('--model', type=str, default='resnet50', help='Model name. Set it resnet50.', required = False)
    parser.add_argument('--epochs', type=int, default=10, help='Define number of training epochs.', required = False)
    parser.add_argument('--dataset-path', type=str, default='', help='Path to folder containing dataset directory.', required = False)
    parser.add_argument('--batch-size', type=int, default=32, help='Define batch size.', required = False)
    parser.add_argument('--target-size', type=int, nargs=2, default=[200, 200], help='Image size for model.', required = False)
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Set learning rate.', required = False)

    return parser.parse_args()