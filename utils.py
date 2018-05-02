import numpy as np
import matplotlib.pyplot as plt


def join_preds(preds, idxs, method=np.sum):
    nb_samples = int(np.max(idxs)+1)
    print (nb_samples, preds.shape)
    ret_preds = np.zeros((nb_samples, preds.shape[1]), dtype=preds.dtype)

    m = {}
    for i, j in enumerate(idxs):
        m.setdefault(j, [])
        m[j].append(i)

    for i in range(nb_samples):
        ret_preds[i] = method(preds[m[i]], axis=0)

    return np.argmax(ret_preds, axis=1)


def plot_model_history(h, mode="loss", file_name=""):
    legend = [mode]

    x = range(len(h[mode]))
    plt.plot(x, h[mode], marker='.')

    if "val_" + mode in h:
        legend.append("val_" + mode)
        plt.plot(x, h["val_" + mode], marker='.')

    plt.xlabel('Epochs')
    plt.title(mode + ' over epochs')
    plt.legend(legend, loc='center right')
    plt.savefig(file_name)
