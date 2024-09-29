

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from flask import Flask, render_template, request
from groq import Groq
import time

app = Flask(__name__)
api_key="your_api_key"
client=Groq(api_key=api_key)


data_path = ('./data/DataSet.xlsx')
df = pd.read_excel(data_path)
ls=[]

for index, row in df.iterrows():
    tId = row["Id"]
    tComponent = row["Component"]
    tParameter = row["Parameter"]
    tValue = int(row["Value"])

    #print(tId,tComponent,tParameter,tValue)

    failProb = "No"
    if(tComponent == "Engine"):
      if((tParameter == "Oil Pressure") and (tValue<25 or tValue>65)):
        failProb = "High"
      elif(tParameter == "Speed" and tValue>1800 ):
        failProb = "Medium"
      elif(tParameter == "Temperature" and tValue>105 ):
        failProb = "High"

    elif(tComponent == "Fuel"):
      if(tParameter == "Water in Fuel" and tValue>1800 ):
        failProb = "High"
      elif(tParameter == "Level" and tValue<1 ):
        failProb = "Low"
      elif(tParameter == "Temperature" and tValue>400 ):
        failProb = "High"
      elif(tParameter == "Pressure" and (tValue<35 or tValue>65 ) ):
        failProb = "Low"

    elif(tComponent == "Drive"):
      if(tParameter == "Transmission Pressure" and (tValue<200 or tValue>450 )):
        failProb = "Medium"
      elif(tParameter == "Brake Control" and tValue<1 ):
        failProb = "Medium"
      elif(tParameter == "Pedal Sensor" and tValue>4.7 ):
        failProb = "Low"

    else:
      if(tParameter == "Exhaust Gas Temparature" and tValue>365 ):
        failProb = "High"
      elif(tParameter == "Air Filter Pressure" and tValue<20 ):
        failProb = "Medium"
      elif(tParameter == "System Voltage" and (tValue<12.0 or tValue>15.0 )):
        failProb = "High"
      elif(tParameter == "Hydraulic Pump Rate" and tValue>125 ):
        failProb = "Medium"
    ls.append(failProb)
df['failProb'] = pd.Series(ls)
X = df[['Machine','Component','Parameter','Value']]
y = df['failProb']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing for text features
text_features = ['Component', 'Parameter']
numeric_features = ['Value']

# Text preprocessing
text_transformer = Pipeline(steps=[
    ('tfidf', TfidfVectorizer())
])

# Numeric preprocessing
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('component', text_transformer, 'Component'),
        ('parameter', text_transformer, 'Parameter'),
        ('value', numeric_transformer, ['Value'])
    ]
)

# Define the model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)


# Evaluate the model
print(classification_report(y_test, y_pred))


# Function to preprocess new data
def preprocess_new_data(machine, component, parameter, value, preprocessor):
    new_data = pd.DataFrame([[machine, component, parameter, value]], columns=['Machine', 'Component', 'Parameter', 'Value'])
    new_data_preprocessed = preprocessor.transform(new_data)
    return new_data_preprocessed

# Function to predict fail probability for a new input row
def predict_failure(machine, component, parameter, value):
    new_data_preprocessed = preprocess_new_data(machine, component, parameter, value, preprocessor)
    prediction = model.named_steps['classifier'].predict(new_data_preprocessed)
    print("new data- ", new_data_preprocessed)
    print("predict", prediction[0])
    return prediction[0]




@app.route('/predict', methods=['POST'])
def predict():
    machine = request.form['machine']
    component = request.form['component']
    parameter = request.form['parameter']
    value = float(request.form['value'])

    predicted_fail_prob = predict_failure(machine, component, parameter, value)
    return render_template('result.html', machine=machine, component=component, parameter=parameter, value=value, prediction=predicted_fail_prob)

@app.route('/', methods=['GET', 'POST'])
def index():
    data_path = './data/RealTimeDataSet.xlsx'
    df = pd.read_excel(data_path)
    results = []
    for index, row in df.iterrows():
        machine = row['Machine']
        component = row['Component']
        parameter = row['Parameter']
        value = float(row['Value'])

        # Predict failure probability
        predicted_fail_prob = predict_failure(machine, component, parameter, value)

        # Print prediction result
        # print(f"Machine: {machine}, Component: {component}, Parameter: {parameter}, Value: {value}, Prediction: {predicted_fail_prob}")
        # Store results to display on the web page
        results.append({
            'machine': machine,
            'component': component,
            'parameter': parameter,
            'value': value,
            'prediction': predicted_fail_prob
        })
        return render_template('result.html', results=results)


# def index():
#     if request.method == 'POST':
#         machine = request.form['machine']
#         component = request.form['component']
#         parameter = request.form['parameter']
#         value = float(request.form['value'])

#         predicted_fail_prob = predict_failure(machine, component, parameter, value)
#         return render_template('result.html', prediction=predicted_fail_prob)

#     return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
