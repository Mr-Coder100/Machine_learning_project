import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)

crops = ['Wheat', 'Rice', 'Maize', 'Cotton', 'Sugarcane', 'Pulses', 'Mustard', 'Soybean', 'Groundnut', 'Barley']
seasons = ['Kharif', 'Rabi', 'Zaid']

data = []
for _ in range(800):
    crop = np.random.choice(crops)
    season = np.random.choice(seasons)
    rainfall = np.random.randint(300, 1500)
    prev_price = np.random.randint(1200, 6000)
    production = np.random.randint(50, 500)
    area = np.random.randint(10, 300)
    export_demand = np.random.randint(0, 100)

    score = 0
    if rainfall > 900: score += 1
    if prev_price > 3000: score += 2
    if production < 150: score += 2
    if export_demand > 60: score += 1
    if area < 80: score += 1

    demand = 'High' if score >= 4 else ('Medium' if score >= 2 else 'Low')
    data.append([crop, season, rainfall, prev_price, production, area, export_demand, demand])

df = pd.DataFrame(data, columns=['Crop','Season','Rainfall','PrevPrice','Production','Area','ExportDemand','Demand'])

le_crop = LabelEncoder()
le_season = LabelEncoder()
le_demand = LabelEncoder()

df['Crop_enc'] = le_crop.fit_transform(df['Crop'])
df['Season_enc'] = le_season.fit_transform(df['Season'])
df['Demand_enc'] = le_demand.fit_transform(df['Demand'])

X = df[['Crop_enc','Season_enc','Rainfall','PrevPrice','Production','Area','ExportDemand']]
y = df['Demand_enc']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

pickle.dump(model, open('demand_model.pkl', 'wb'))
pickle.dump({'crop': le_crop, 'season': le_season, 'demand': le_demand}, open('demand_encoders.pkl', 'wb'))
print("Done!")