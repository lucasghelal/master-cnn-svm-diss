# Cria um arquivo TXT com as features e labels para rodar no svm
# saida: file
# entradas:
# features = [
#   [feature1, feature2, feature3],
#   [feature1, feature2, feature3]
# ]
# labels = [
#   label1,
#   label2,
# ]


def createTxt(file, features, labels):
    with open(file, 'w') as f:
        for feature, label in zip(features, labels):
            f.write(' '.join(feature) + ' ' + label + "\n")
