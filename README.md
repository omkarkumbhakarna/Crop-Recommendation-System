import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Read the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Exploratory Data Analysis (EDA)
print(data.head())
print(data.tail())
print(data.shape)
print(data.columns)
print(data.duplicated().sum())
print(data.isnull().sum())
print(data.info())
print(data.describe())
print(data.nunique())
print(data['label'].unique())
print(data['label'].value_counts())

# Summary statistics for each crop
crop_summary = data.groupby('label').mean()

# Box plots for each feature
for i in data.columns:
    plt.figure(figsize=(15,6))
    sns.boxplot(data[i])
    plt.xticks(rotation=90)
    plt.show()

# Bar plots for N, P, and K values for each crop
crop_summary_new = crop_summary.reset_index()

plt.figure(figsize=(15,6))
sns.barplot(y='N', x='label', data=crop_summary_new, palette='hls')
plt.xticks(rotation=90)
plt.show()

fig1 = px.bar(crop_summary_new, x='label', y='N')
fig1.show()

plt.figure(figsize=(15,6))
sns.barplot(y='P', x='label', data=crop_summary_new, palette='hls')
plt.xticks(rotation=90)
plt.show()

fig2 = px.bar(crop_summary_new, x='label', y='P')
fig2.show()

plt.figure(figsize=(15,6))
sns.barplot(y='K', x='label', data=crop_summary_new, palette='hls')
plt.xticks(rotation=90)
plt.show()

fig3 = px.bar(crop_summary_new, x='label', y='K')
fig3.show()

# Visualization for N, P, and K values comparison between crops
# This part of the code is already correct

# Visualization for NPK ratio for rice, cotton, jute, maize, and lentil
# This part of the code is already correct

# Scatter plot for temperature vs humidity colored by crop label
crop_scatter = data[(data['label']=='rice') | (data['label']=='jute') | (data['label']=='cotton') | (data['label']=='maize') | (data['label']=='lentil')]

fig = px.scatter(crop_scatter, x="temperature", y="humidity", color="label", symbol="label")
fig.update_layout(plot_bgcolor='yellow')
fig.update_xaxes(showgrid=True)
fig.update_yaxes(showgrid=True)
fig.show()

# Bar plot for comparing rainfall, temperature, and humidity across crops
fig = px.bar(crop_summary, x=crop_summary.index, y=["rainfall", "temperature", "humidity"])
fig.update_layout(title_text="Comparison between rainfall, temperature, and humidity", plot_bgcolor='white', height=500)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()

# Correlation matrix
plt.figure(figsize=(15, 9))
sns.heatmap(data.corr(), annot=True, cmap='Wistia')
plt.xlabel('Features')
plt.ylabel('Features')
plt.title('Correlation between different features')
plt.show()

# Machine Learning Models
X = data.drop(['label'], axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = svm.SVC(kernel='linear', C=1, gamma='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("SVM Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False, square=True, xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title('Confusion matrix')
plt.show()

# Decision Tree Classifier
classifier_dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier_dt.fit(X_train, y_train)
y_pred_dt = classifier_dt.predict(X_test)
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print('Decision Tree Model accuracy score:', accuracy_dt)
print(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt)
plt.figure(figsize=(7, 7))
sns.heatmap(cm_dt, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix - Decision Tree')
plt.show()

# Random Forest Classifier
classifier_rf = RandomForestClassifier(n_estimators=10, criterion="entropy")
classifier_rf.fit(X_train, y_train)
y_pred_rf = classifier_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print('Random Forest Model accuracy score:', accuracy_rf)
print(classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(7, 7))
sns.heatmap(cm_rf, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix - Random Forest')
plt.show()
