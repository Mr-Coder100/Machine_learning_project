from flask import Flask, render_template, flash,redirect
import joblib
import os
from flask import request
import numpy as np
import pickle
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for
from weather import Weather, WeatherException

app = Flask(__name__)

app.config.from_pyfile('config/config.cfg')
w = Weather(app.config)

model = pickle.load(open('yield.pkl', 'rb'))
crop = pickle.load(open('crop.pkl', 'rb'))
fertilizer_model    = pickle.load(open('fertilizer.pkl', 'rb'))
fertilizer_encoders = pickle.load(open('fertilizer_encoders.pkl', 'rb'))

@app.route('/')

@app.route('/index') 
def index():
	return render_template('index.html')
@app.route('/login') 
def login():
	return render_template('login.html')    
@app.route('/chart') 
def chart():
	return render_template('chart.html')
 
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
@app.route('/upload') 
def upload():
	return render_template('upload.html') 
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

@app.route("/fertilizer")
def fertilizer():
    return render_template("fertilizer.html")

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
