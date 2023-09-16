
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('Crop_recommendation.csv')


X = data.drop('label', axis=1)
y = data['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


clf = DecisionTreeClassifier()


clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


new_data = pd.DataFrame({
    'temperature': [25],
    'humidity': [70],
    'rainfall': [100],
    'pH': [6.5],
    'N':[100],
    'P':[29],
    'K':[40]
})


recommended_crop = clf.predict(new_data)
print(f'Recommended Crop: {recommended_crop[0]}')
