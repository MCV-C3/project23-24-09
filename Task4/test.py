import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle


if __name__ == '__main__':
    # Load the model
    model = keras.models.load_model('model-best.h5')

    # Load test dataset
    test_dataset = keras.preprocessing.image_dataset_from_directory(
        directory='../MIT_split/test',
        labels='inferred',
        label_mode='categorical',
        batch_size=32,
        class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
        image_size=(64, 64),
        shuffle=False,
        validation_split=None,
        subset=None
    )

    preprocessing_test = keras.Sequential([
        keras.layers.experimental.preprocessing.Rescaling(1. / 255)
    ])

    test_dataset = test_dataset.map(lambda x, y: (preprocessing_test(x, training=False), y))
    test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Evaluate the model
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    # Predict the test dataset
    y_pred = model.predict(test_dataset)
    y_pred = np.argmax(y_pred, axis=1)
    y_true = np.concatenate([y for x, y in test_dataset], axis=0)
    y_true = np.argmax(y_true, axis=1)

    count = 0
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            count += 1

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    class_names = ['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding']

    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot confusion matrix
    map = sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    map.set_xticklabels(map.get_xticklabels(), rotation=45, horizontalalignment='right')
    map.set_yticklabels(map.get_yticklabels(), rotation=0, horizontalalignment='right')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Plot ROC curve

    n_classes = 8
    lw = 2

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_score = model.predict(test_dataset)
    y_test = np.concatenate([y for x, y in test_dataset], axis=0)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure(figsize=(8, 8))
    lw = 2
    plt.plot(fpr["macro"], tpr["macro"],
             label='average ROC curve (auc = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    palette = sns.color_palette("hls", 8)
    colors = cycle(palette)
    for i, color, name in zip(range(n_classes), colors, class_names):
        label = 'Class {0} ROC curve (auc = {1:0.2f})'.format(name, roc_auc[i])
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label=label)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for each of the classes')
    plt.legend(loc="lower right")
    plt.show()


