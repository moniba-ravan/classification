import sys


from models import load_model
from params import get_args
from data.data_loader import get_loader

from keras.optimizers import Adam


def train():
    model_name = sys.argv[2]
    print(f"Chosen Model: {model_name}")
    args = get_args(model_name)
    print(f"Arguments: {args}")
    train_loader, val_loader, test_loader = get_loader(args.dataset_dir)
    model = load_model(model_name=model_name, **args)

    # training
    opt = Adam(lr=0.0001)
    model.compile(optimizer=opt,  loss= 'categorical_crossentropy', metrics=['acc'])
    history = model.fit(train_loader, steps_per_epoch=100, epochs=20,
                       validation_data=val_loader, validation_steps=100)


if __name__ == '__main__':
    train()
