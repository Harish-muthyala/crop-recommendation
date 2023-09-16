
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('crop_data.csv')


X = data.drop('Crop', axis=1)
y = data['Crop']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = DecisionTreeClassifier()


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


new_data = pd.DataFrame({
    'Temperature': [25],
    'Humidity': [70],
    'Rainfall': [100],
    'pH Level': [6.5]
})


recommended_crop = clf.predict(new_data)
print(f'Recommended Crop: {recommended_crop[0]}')
