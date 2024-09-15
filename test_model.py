from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = load_iris()
X = data.data  
y = data.target  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def train():
    model = LogisticRegression(max_iter=150, random_state=42)
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)


    accuracy = accuracy_score(y_test, y_pred)


    print(f'Model Accuracy: {accuracy:.4f}')
    return accuracy,y_pred

def test_model_accuracy():
    accuracy = train()
    # Assert that the accuracy is above a certain threshold, e.g., 90%
    assert accuracy >= 0.9