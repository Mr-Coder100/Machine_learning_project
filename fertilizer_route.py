# =====================================================================
# FERTILIZER RECOMMENDER — Add this code to your existing app.py
# =====================================================================
# STEP 1: Add these imports at the top of app.py (if not already present)
# from flask import Flask, render_template, request
import pickle
import numpy as np

# STEP 2: Load models — add alongside your existing crop.pkl / yield.pkl loads
fertilizer_model = pickle.load(open('fertilizer.pkl', 'rb'))
fertilizer_encoders = pickle.load(open('fertilizer_encoders.pkl', 'rb'))

# STEP 3: Add this route to app.py
# @app.route('/fertilizer', methods=['GET', 'POST'])
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


# =====================================================================
# FULL app.py EXAMPLE — How your app.py should look after integration
# =====================================================================
"""
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load all models
crop_model          = pickle.load(open('crop.pkl', 'rb'))
yield_model         = pickle.load(open('yield.pkl', 'rb'))
fertilizer_model    = pickle.load(open('fertilizer.pkl', 'rb'))
fertilizer_encoders = pickle.load(open('fertilizer_encoders.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# --- Your existing crop & yield routes here ---

@app.route('/fertilizer', methods=['GET', 'POST'])
def fertilizer():
    result = None
    error  = None
    if request.method == 'POST':
        try:
            temperature = float(request.form['temperature'])
            humidity    = float(request.form['humidity'])
            moisture    = float(request.form['moisture'])
            soil_type   = request.form['soil_type']
            crop_type   = request.form['crop_type']
            nitrogen    = float(request.form['nitrogen'])
            potassium   = float(request.form['potassium'])
            phosphorous = float(request.form['phosphorous'])

            soil_enc = fertilizer_encoders['soil'].transform([soil_type])[0]
            crop_enc = fertilizer_encoders['crop'].transform([crop_type])[0]

            features = np.array([[temperature, humidity, moisture,
                                   soil_enc, crop_enc,
                                   nitrogen, potassium, phosphorous]])

            pred_encoded = fertilizer_model.predict(features)[0]
            result = fertilizer_encoders['fertilizer'].inverse_transform([pred_encoded])[0]
        except Exception as e:
            error = str(e)

    return render_template('fertilizer.html', result=result, error=error)

if __name__ == '__main__':
    app.run(debug=True)
"""
