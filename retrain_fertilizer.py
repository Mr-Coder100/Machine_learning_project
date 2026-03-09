import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('Fertilizer Prediction.csv')
df.columns = df.columns.str.strip()

le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fert = LabelEncoder()

df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])
df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])
df['Fertilizer Name'] = le_fert.fit_transform(df['Fertilizer Name'])

X = df.drop('Fertilizer Name', axis=1)
y = df['Fertilizer Name']

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

pickle.dump(model, open('fertilizer.pkl', 'wb'))
pickle.dump({'soil': le_soil, 'crop': le_crop, 'fertilizer': le_fert}, open('fertilizer_encoders.pkl', 'wb'))
print("Done! fertilizer.pkl retrained successfully!")