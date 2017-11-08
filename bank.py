import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow.contrib.learn as learn
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
plt.show()


#Receiving the Data
data = pd.read_csv('bank_note_data.csv')
data.head()
sns.countplot(x='Class',data=data,palette='coolwarm')
sns.pairplot(data,hue='Class',palette='coolwarm')
# Scaling
scaler = StandardScaler()
scaler.fit(data.drop('Class',axis=1))
scaled_features = scaler.fit_transform(data.drop('Class',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_feat.head()

#Train Test Split
X = df_feat
y = data['Class']
X = X.as_matrix()
y = y.as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2)
classifier.fit(X_train, y_train, steps=200, batch_size=20)

#Model Evaluation
note_predictions = classifier.predict(X_test)
print(confusion_matrix(y_test,note_predictions))
print(classification_report(y_test,note_predictions))

#Comparison
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train,y_train)
rfc_preds = rfc.predict(X_test)
print(classification_report(y_test,rfc_preds))
print(confusion_matrix(y_test,rfc_preds))
