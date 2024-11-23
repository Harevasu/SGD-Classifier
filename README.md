# SGD-Classifier
## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### Algorithm

1. **Import necessary libraries.**  
2. **Load the Iris dataset.**  
3. **Convert the dataset into a pandas DataFrame and inspect the first few rows.**  
4. **Separate the features (`X`) and target variable (`y`).**  
5. **Split the data into training and testing sets.**  
6. **Initialize the `SGDClassifier` with appropriate parameters.**  
7. **Train the classifier on the training data.**  
8. **Make predictions on the test data.**  
9. **Evaluate the model using accuracy score.**  
10. **Compute and display the confusion matrix.**

## Program:

### Program to implement the prediction of iris species using SGD Classifier.
### Developed by: HAREVASU S
### RegisterNumber:  212223230069

```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

print(df.head())

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(X_train, y_train)

y_pred = sgd_clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```

## Output:
![Screenshot 2024-09-21 095242](https://github.com/user-attachments/assets/43164337-4cfd-4c2c-971c-01175e8f88dd)



## Result:
Thus, the program to implement the prediction of the Iris species using SGD Classifier is written and verified using Python programming.
