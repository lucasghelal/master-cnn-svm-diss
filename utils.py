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


def write_svm(nome, features, labels):
    with open(nome, 'w') as f:
        for label, features_line in zip(labels, features):
            features_str = []
            for i, feature in enumerate(features_line):
                if feature != 0:
                    features_str.append('%d:%.6f' % (i+1, feature))
            f.write("%d %s\n" % (label, ' '.join(features_str)))


def gerar_svm(X, Y, num_blocos, get_output, mode=np.sum):
    features = []
    labels = []
    for i in range(0, X.shape[0], num_blocos):
        out = get_output([X[i:i+num_blocos], 0])
        final = mode(out[0], axis=0)
        features.append(final.tolist())
        labels.append(np.argmax(Y[i])+1)
    return labels, features
