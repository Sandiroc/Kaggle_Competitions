import numpy as np
import pandas as pd
import sklearn as sk
from sklearn import metrics
import os
import matplotlib.pyplot as plt
import seaborn as sns

def format_data():
    df = pd.read_csv("data/train.csv")
    df = df[['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    return df

def correlations(df):
        x = df['Sex']
        y = df['Survived']
        phi = metrics.matthews_corrcoef(x, y)
        matrix = metrics.confusion_matrix(x, y)

        plt.figure(figsize=(8,6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='coolwarm', 
                    xticklabels=['Male', 'Female'], yticklabels= ['Yes', 'No'])
        plt.title('Sex vs Survived: Phi = ' + str("{:.2f}".format(phi)))
        plt.xlabel('Sex')
        plt.ylabel('Survived')
        plt.savefig('plot/sex_vs_survived.png')

        
        x = df['Age']
        plt.figure(figsize=(8,6))
        sns.boxplot(x='Survived', y='Age', data=df)
        plt.xlabel('Survived?')
        plt.ylabel('Age')

        plt.savefig('plot/age_vs_survived.png')


if __name__ == "__main__":
    df = format_data()
    print(df)
    correlations(df)

