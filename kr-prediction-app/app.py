# it brings in different packages that allows to make a line
import pandas as pd 
import numpy as np # allows me to make arrays and math
import matplotlib.pyplot as plt # basic charts/visualization 

# these are my ML models, this is what is predicting everything
from sklearn.linear_model import LinearRegression # makes my line
from sklearn.preprocessing import PolynomialFeatures # transforms my (x) to (x^2, x^3, etc..)
from sklearn.pipeline import make_pipeline # brings those steps together

# loads my excel file
df = pd.read_excel("RateConstantEx.xlsx")

# extract data from ecel file and plot
ph_levels = df["pH"].values.reshape(-1,1)
kr_values = df["Kr"].values


degree = 3 # how curvy the line is
model = make_pipeline(PolynomialFeatures(degree), LinearRegression()) # Puts in the inputs to make the line curvy
model.fit(ph_levels, kr_values) #this is where the model finds the best curve

ph_range = np.linspace(0, 14, 100).reshape(-1, 1) # makes it a smooth curve not just a few dots
kr_predictions = model.predict(ph_range) # uses model to predict Kr values

import streamlit as st

st.title("pH vs Kr Predictor")

ph = st.slider("Select a pH level:", 0.0, 14.0, 7.0, step = .1)
predicted_kr = model.predict([[ph]])[0]
st.write(f"Predicted Kr for pH = {ph:.2f}: **{predicted_kr:.6f}**")

# makes it so we can see our graph
plt.scatter(ph_levels, kr_values, color = 'blue', label = 'Actual Data')
plt.plot(ph_range, kr_predictions, color = 'red', label = 'Linear Regression Line (0-14)')
plt.scatter(ph, predicted_kr, color = 'green', label = f'Predicted at pH = {ph}')
plt.xlabel('pH Level')
plt.ylabel('Rate Constant Kr (min^-1)')
plt.title('pH vs Kr Prediction (Realistic Range)')
plt.legend()
plt.grid(True)
plt.xlim(0,14)
plt.tight_layout()
st.pyplot(plt)