from flask import Flask, render_template, flash,redirect
import joblib
import os
from flask import request
import numpy as np
import pickle
import pandas as pd

from flask import Flask, render_template, flash, redirect, request, url_for, jsonify, session
from weather import Weather, WeatherException
import requests

app = Flask(__name__)
app.secret_key = 'agroguide_secret_key'

app.config.from_pyfile('config/config.cfg')
w = Weather(app.config)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, 'yield.pkl'), 'rb'))
crop = pickle.load(open(os.path.join(BASE_DIR, 'crop.pkl'), 'rb'))
fertilizer_model = pickle.load(open(os.path.join(BASE_DIR, 'fertilizer.pkl'), 'rb'))
fertilizer_encoders = pickle.load(open(os.path.join(BASE_DIR, 'fertilizer_encoders.pkl'), 'rb'))
demand_model = pickle.load(open(os.path.join(BASE_DIR, 'demand_model.pkl'), 'rb'))
demand_encoders = pickle.load(open(os.path.join(BASE_DIR, 'demand_encoders.pkl'), 'rb'))

@app.route('/')
def home():
    return redirect(url_for('index'))


# ============================================================
# ROUTE 1: Crop Investment Estimator
# ============================================================
@app.route('/investment_estimator')
def investment_estimator():
    return render_template('investment_estimator.html')


# ============================================================
# ROUTE 2: Fair Price Estimator
# ============================================================
@app.route('/fair_price')
def fair_price():
    return render_template('fair_price.html')


# ============================================================
# ROUTE 3: Investment vs Profit Calculator
# ============================================================
@app.route('/profit_calculator')
def profit_calculator():
    return render_template('profit_calculator.html')


# ============================================================
# ROUTE 4: Demand Forecasting (page + API)
# ============================================================
@app.route('/demand_forecast', methods=['GET', 'POST'])
def demand_forecast_page():
    if request.method == 'POST':
        try:
            data = request.get_json()
            crop_name = data['crop']
            season = data['season']
            rainfall = float(data['rainfall'])
            prev_price = float(data['prev_price'])
            production = float(data['production'])
            area = float(data['area'])
            export_demand = float(data['export_demand'])

            crop_enc = demand_encoders['crop'].transform([crop_name])[0]
            season_enc = demand_encoders['season'].transform([season])[0]

            features = np.array([[crop_enc, season_enc, rainfall, prev_price, production, area, export_demand]])
            pred = demand_model.predict(features)[0]
            result = demand_encoders['demand'].inverse_transform([pred])[0]

            return jsonify({'demand': result})
        except Exception as e:
            return jsonify({'demand': 'Medium', 'error': str(e)})
    return render_template('demand_forecast.html')

@app.route('/index') 
def index():
	return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])

def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username == 'MrCoder100' and password == 'msit1234':
            session['logged_in'] = True
            session['username'] = 'Lokesh'
            return redirect(url_for('index'))
        else:
            error = 'Invalid credentials!'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/chart') 
def chart():
	return render_template('chart.html')


# 2. Add your NewsAPI key (get free key from https://newsapi.org)
NEWS_API_KEY = "2001e039d621410785e82413e47a2d9c"  # Replace with your key


# ============================================================
# ROUTE 1: Economic Analysis
# ============================================================
@app.route('/economic')
def economic():
    articles = []
    news_error = False
    try:
        if NEWS_API_KEY != "YOUR_NEWSAPI_KEY_HERE":
            url = f"https://newsapi.org/v2/everything?q=india+farmer+subsidy+scheme&language=en&sortBy=publishedAt&pageSize=6&apiKey={NEWS_API_KEY}"
            response = requests.get(url, timeout=5)
            data = response.json()
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
    except Exception:
        news_error = True
    return render_template('economic.html', articles=articles, news_error=news_error)


# ============================================================
# ROUTE 2: Sustainable Farming
# ============================================================
@app.route('/sustainable')
def sustainable():
    return render_template('sustainable.html')


# ============================================================
# ROUTE 3: Farmer Events & Information
# ============================================================
@app.route('/events')
def events():
    articles = []
    try:
        if NEWS_API_KEY != "YOUR_NEWSAPI_KEY_HERE":
            url = f"https://newsapi.org/v2/everything?q=india+agriculture+farmer+news&language=en&sortBy=publishedAt&pageSize=9&apiKey={NEWS_API_KEY}"
            response = requests.get(url, timeout=5)
            data = response.json()
            if data.get('status') == 'ok':
                articles = data.get('articles', [])
    except Exception:
        pass
    return render_template('farmer_events.html', articles=articles)

 
@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer():
    result = None
    error = None

    if request.method == 'POST':
        try:
            # Collect form inputs
            temperature  = float(request.form['temperature'])
            humidity     = float(request.form['humidity'])
            moisture     = float(request.form['moisture'])
            soil_type    = request.form['soil_type']
            crop_type    = request.form['crop_type']
            nitrogen     = float(request.form['nitrogen'])
            potassium    = float(request.form['potassium'])
            phosphorous  = float(request.form['phosphorous'])

            # Encode categorical values
            soil_enc  = fertilizer_encoders['soil'].transform([soil_type])[0]
            crop_enc  = fertilizer_encoders['crop'].transform([crop_type])[0]

            # Prepare input array (same column order as training)
            features = np.array([[temperature, humidity, moisture,
                                   soil_enc, crop_enc,
                                   nitrogen, potassium, phosphorous]])

            # Predict
            pred_encoded = fertilizer_model.predict(features)[0]
            result = fertilizer_encoders['fertilizer'].inverse_transform([pred_encoded])[0]

        except ValueError as e:
            error = f"Invalid input: {e}"
        except Exception as e:
            error = f"Error: {e}"

    return render_template('fertilizer.html', result=result, error=error)   
# @app.route('/upload') 
# def upload():
# 	return render_template('upload.html') 
@app.route('/preview',methods=["POST"])
def preview():
    if request.method == 'POST':
        dataset = request.files['datasetfile']
        df = pd.read_csv(dataset,encoding = 'unicode_escape')
        df.set_index('Id', inplace=True)
        return render_template("preview.html",df_view = df)    

 
@app.route('/yield_prediction')
def yield_prediction():
    return render_template('yield_prediction.html')




@app.route("/predict",methods = ["POST"])
def predict():
    if request.method == "POST":
        print(request.form)
        State_Name = request.form['State_Name']
        Crop = request.form['Crop']
        Area = request.form['Area']
        Soil_type = request.form['soil_type']
         
        pred_args = [State_Name,Crop,Area,Soil_type]
        pred_args_arr = np.array(pred_args)
        pred_args_arr = pred_args_arr.reshape(1,-1)
        output = model.predict(pred_args_arr)
        print(output)
        pred=format(int(output[0]))
        Yield= int(pred) / float(Area)
        yields= Yield

    return render_template("yield_prediction.html",prediction_text=pred, yield_predictions= int(yields))
@app.route('/crop_prediction')
def crop_prediction():
    return render_template('crop_prediction.html')

@app.route('/sandy',methods=['POST'])
 
def sandy():
  
 

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    
    prediction =  crop.predict(final_features)


    preds=format((prediction[0]))

    return render_template("crop_prediction.html",prediction_texts=preds)
    
@app.route('/homepage')
def homepage():
    return render_template('homepage.html')

@app.route('/result', methods=['POST', 'GET'])
def result_page():
    if request.method == 'POST':
        location = request.form
        w.set_location(location.get('location'))

        try:
            return render_template('result.html', data=w.get_forecast_data())
        except WeatherException:
            app.log_exception(WeatherException)
            return render_template('error.html')
    else:
        return redirect(url_for('homepage'))    
if __name__ == "__main__":
    
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
