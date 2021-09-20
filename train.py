import sys


from models import load_model
from params import get_args
from data.data_loader import get_loader
from keras.optimizers import adam_v2 as Adam

def train():
    # model_name = sys.argv[2]
    # # model_name = 'resnet50'
    # print(f"Chosen Model: {model_name}")
    # args = get_args(model_name)
    # print(f"Arguments: {args}")


    # Loading Data
    train_loader, valid_loader, test_loader = get_loader(args.dataset_path,  # dataset dir path
                                                         args.batch_size,
                                                         tuple(args.target_size)
                                                         # image size that generated by data loader
                                                         )
    print("Loading Data is Done!")

    # Loading model
    model = load_model(model_name=model_name,
                       image_size=args.target_size
                       )
    print("Loding Model is Done!")

    # training
    opt = Adam(learning_rate=args.learning_rate)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  metrics=['acc']
                  )
    print("Training Model...")
    history = model.fit(train_loader,
                        steps_per_epoch=train_loader.n // args.batch_size,
                        epochs=args.epochs,
                        validation_data=valid_loader,
                        validation_steps=valid_loader.n // args.batch_size
                        )
    print("Training Model is Done!")

if __name__ == '__main__':
    train()
