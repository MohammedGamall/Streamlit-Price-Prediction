import joblib
import numpy 
import pandas 
import streamlit 
import seaborn 
import plotly.express 
import matplotlib.pyplot
import sklearn


df = pandas.read_csv('final_df.csv')
model = joblib.load("random_forest.pkl")

print(joblib.__version__)
print(numpy.__version__)
print(pandas.__version__)
print(seaborn.__version__)
print(plotly.__version__)
print(matplotlib.__version__)
print(sklearn.__version__)
print(python.__version)

