#knn, svm, tree

from matplotlib.pylab import cast
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df_red = pd.read_csv('C:\\Users\\andre\\Desktop\\Sandbox\\Data Mining Classification\\winequality-red.csv', sep=';')
df_white = pd.read_csv('C:\\Users\\andre\\Desktop\\Sandbox\\Data Mining Classification\\winequality-white.csv', sep=';')

cast = {
    "fixed acidity": float,
    "volatile acidity": float,
    "citric acid": float,
    "residual sugar": float,
    "chlorides": float,
    "free sulfur dioxide": float,
    "total sulfur dioxide": float,
    "density": float,
    "pH": float,
    "sulphates": float,
    "alcohol": float,
    "quality": float
}

df_red = df_red.astype(cast)
df_white = df_white.astype(cast)

print(df_red.dtypes)
print(df_white.dtypes)

#Index(['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'], dtype='str')
#Index(['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"'], dtype='str')