from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .mlflow_handler import MLFlowHandler


def get_plots(model, test_loader, args, mlflow_handler: MLFlowHandler):
    # Metrics: Test: Loss, Acc
    n_classes = args.n_classes
    print('Evaluation')
    predictions = model.predict(test_loader, steps=test_loader.n // args.batch_size + 1)
    y_pred = np.argmax(predictions, axis=-1)
    y_true = test_loader.classes

    test_score = model.evaluate(test_loader, steps=test_loader.n // args.batch_size + 1)  # test data
    print(f'Test: loss= {test_score[0]}, Accuracy: {test_score[1]}')

    # Metrics: Confusion Matrix
    con_mat = confusion_matrix(y_true, y_pred)
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=[i for i in range(n_classes)], columns=[i for i in range(n_classes)])
    figure = plt.figure(figsize=(n_classes, n_classes))
    sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('confusion matrix')
    mlflow_handler.add_figure(figure, 'images/confusion_matrix.png')
