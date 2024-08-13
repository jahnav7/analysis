from flask import Flask, make_response, request, render_template
import io
from io import StringIO
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

def feature_engineering(df):
    df = df.drop(['CustomerId','Surname'], axis=1)
    df['Gender'] = np.where(df['Gender'] == " Male", 1, 0)
    
    label_enco_geo = {value: key for key, value in enumerate(df['Geography'].unique())}
    df['Geography'] = df['Geography'].map(label_enco_geo)
    return df

def scalar(df):
    sc = StandardScaler()
    X = df[['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']]
    X = sc.fit_transform(X)
    return (X)


@app.route('/', methods=['GET'])
def Home():
    return render_template('index.html')


@app.route('/predict', methods=["POST"])
def predict():
    f = request.files['data_file']
    if not f:
        return render_template('index.html', prediction_text="No file selected")

    stream = io.StringIO(f.stream.read().decode("UTF8"), newline=None)
    result = stream.read()#.replace("=", ",")
    df = pd.read_csv(StringIO(result))
    
    #Feature Engineering
    df = feature_engineering(df)
    
    X = scalar(df)
    
    # load the model from disk
    loaded_model = pickle.load(open("rf_model.pkl", 'rb'))
    
    print (loaded_model)

    result = loaded_model.predict(X)
    
    return render_template('index.html', prediction_text="Predicted Salary is/are: {}".format(result))

if __name__ == "__main__":
    app.run(debug=False,port=5000)