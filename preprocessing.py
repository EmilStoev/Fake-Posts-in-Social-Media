import string
import numpy as np

def remove_punct(tweetText):
    all = [ch for ch in tweetText if ch not in string.punctuation]
    clear = ''.join(all)
    return clear

# To provide visualisation I will adapt the code for Confusion Matrix from sklearn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import itertools
import matplotlib.pyplot as plt

def plot_conf_matrix(cm, classes,
                     title='',
                     cmap=plt.cm.Greens):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(2)
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 color='white' if cm[i, j] > thresh else 'black',
                 horizontalalignment='center')
    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')