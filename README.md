import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

file_path = 'car_data_1000.csv'  # Укажите путь к вашему файлу car.csv
data = pd.read_csv(file_path)

print(data.head())

target_variable = 'speed'

features = ['position_x', 'position_y', 'orientation', 'direction', 'speed', 'traffic_light_state', 'traffic_density']
X = data[features]
y = data[target_variable]

X = pd.get_dummies(X, columns=['orientation', 'traffic_light_state', 'traffic_density'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy}")

model_file_path = 'xgboost_model.joblib'

joblib.dump(model, model_file_path)
