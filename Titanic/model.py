import numpy
import pandas as pd
import sklearn as sk

def format_data():
    df = pd.read_csv("data/train.csv")
    df = df[['Sex', 'Age', 'SibSp', 'Parch', 'Fare']]

    print(df.head())
    


if __name__ == "__main__":
    format_data()

