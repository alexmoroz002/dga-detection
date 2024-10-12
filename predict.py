from train import CustomClassifier
import pandas as pd

def main():
    df = pd.read_csv('datasets/test.csv').dropna()
    classifier = CustomClassifier()
    classifier.load()
    pred = classifier.predict(df.domain)
    df['is_dga'] = pred
    df.to_csv('prediction.csv', index=False)
 
if __name__ == "__main__":
    main()