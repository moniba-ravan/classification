from argparse import ArgumentParser


def resnet50_args():
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, default='resnet50', help='Model name. Set it resnet50.', required=False)
    parser.add_argument('--epochs', type=int, default=10, help='Define number of training epochs.', required=False)
    parser.add_argument('--dataset-path', type=str, default='data', help='Path to folder containing dataset directory.',
                        required=False)
    parser.add_argument('--valid-size', type=int, default=0.3, help='Define validation size.', required=False)
    parser.add_argument('--batch-size', type=int, default=32, help='Define batch size.', required=False)
    parser.add_argument('--target-size', type=int, nargs=2, default=[224, 224], help='Image size for model.',
                        required=False)
    parser.add_argument('--n-classes', type=int, default=4, help='Define number of classes', required=False)
    parser.add_argument('--fine_tune', type=bool, default=False, help='To Fine-tune Set True', required=False)
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Set learning rate.', required=False)
    parser.add_argument('--model_path', type=str, default="weights", help='model path', required=False)

    return parser.parse_args()
