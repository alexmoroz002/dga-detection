from train import CustomClassifier
import pandas as pd
from sklearn.metrics import confusion_matrix
from textwrap import dedent


def main():
    df = pd.read_csv('datasets/val.csv').dropna()
    classifier = CustomClassifier()
    classifier.load()
    pred = classifier.predict(df.domain)

    tn, fp, fn, tp = confusion_matrix(df.is_dga, pred).ravel()
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = (2 * precision * recall) / (precision + recall)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    with open('validation.txt', 'w') as f:
        f.write(dedent(f"""\
                    True positive: {tp}
                    False positive: {fp}
                    False negative: {fn}
                    True negative: {tn}
                    Accuracy: {accuracy}
                    Precision: {precision}
                    Recall: {recall}
                    F1: {f1}""")
        )


if __name__ == '__main__':
    main()