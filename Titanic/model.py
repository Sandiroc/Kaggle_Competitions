import numpy
import pandas as pd

def format_data():
    df = pd.read_csv("data/train.csv")
    df = df[['Sex', 'Age', 'SibSp', 'parch', '']]

    


if __name__ == "__main__":
    format_data()

