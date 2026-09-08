# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

"""**Customer Churn Prediction
**
"""

df=pd.read_csv('/content/drive/MyDrive/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.shape

df.head()
pd.set_option("display.max_column",None)



df.head()

df.isnull().sum()

df.info()

df.head()

for col in df.columns:
  print(col,df[col].unique())

numerical_features=["tenure","MonthlyCharges","TotalCharges"]
for col in df.columns:
  if col not in numerical_features:
    print(col,df[col].unique())

# Convert total charges value object->floating object

len(df[df["TotalCharges"]==" "]) # count how many rows where total charges has white shace.It does not have NaN

# Handling Missing Values
df["TotalCharges"]=df["TotalCharges"].replace({" ":"0.0"})
len(df[df["TotalCharges"]==" "])

df["TotalCharges"]=df["TotalCharges"].astype(float)

df["TotalCharges"].dtype
df["Churn"].name

df=df.drop(columns=['customerID'])

"""1. Removed customerid because not required for modelling
2. No missing values in the dataset
3.Missing values in the TotalCharges columns are replaced with 0
4. Class Imbalance identified in the dataset
"""

# Checking the class distribution of Target column
print(df["Churn"].value_counts())

df.describe()

"""Numerical Features analysis"""

from matplotlib.lines import lineStyles
def plot_histogram(df,column_name):
  plt.figure(figsize=(5,3))
  sns.histplot(df[column_name],kde=True)
  plt.title(f"Distribution of {column_name}")

  col_mean=df[column_name].mean()
  col_median=df[column_name].median()

  plt.axvline(col_mean,color="red",linestyle="dashed",linewidth=2,label="Mean")
  plt.axvline(col_median,color="green",linestyle="dashed",linewidth=2,label="Median")

  plt.legend()
  plt.show()

plot_histogram(df,"tenure")

plot_histogram(df,"MonthlyCharges")

plot_histogram(df,"TotalCharges")

# Correlation matrix heatmap
plt.figure(figsize=(8,4))
sns.heatmap(df[["tenure","MonthlyCharges","TotalCharges"]].corr(),annot=True,cmap="coolwarm",fmt=" .2f")
plt.title("Correlation Heatmap")
plt.show()

df.columns

# CountPlot for categorical values
object_cols=df.select_dtypes(include="object").columns.to_list()
object_cols=["SeniorCitizen"]+object_cols
object_cols

for col in object_cols:
  plt.figure(figsize=(5,3))
  sns.countplot(x=df[col])
  plt.title(f"Count Plot for {col}")
  plt.show()

"""**Data Preprocessing**"""

# Label Encoding Categorical Values
# Label encoding of target variable
df["Churn"]=df["Churn"].replace({"Yes":1,"No":0})

df.head()

print(df["Churn"].value_counts())

"""Lable Encoding for catergorical values"""

# Identifying columns with object datatype
object_columns=df.select_dtypes(include="object").columns
print(object_columns)
for col in object_columns:
  print(col)

#initialize a dictionary to save the encoders
encoders={}

# apply label encoding and store the encoders
for column in object_columns:
  label_encoder=LabelEncoder()
  df[column]=label_encoder.fit_transform(df[column])
  encoders[column]=label_encoder

# save the encoders to a pickle file
with open("encoders.pkl","wb") as f:
  pickle.dump(encoders,f)

"""Training and test data split"""

X=df.drop(columns=["Churn"])

print(X)

y=df["Churn"]

#split training and test data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

print(y_train.value_counts())

"""Due to imbalance in target variable we have to use oversampling SMOTE technique"""

smote=SMOTE(random_state=42)

X_train_smote,y_train_smote=smote.fit_resample(X_train,y_train)
print(y_train_smote.shape)
print(y_train_smote.value_counts())

"""**Model Training**"""

#Training  with default hyperparameters

models={
    "DecisionTreeClassifier":DecisionTreeClassifier(random_state=42),
    "RandomForestClassifier":RandomForestClassifier(random_state=42),
    "XGBClassifier":XGBClassifier(random_state=42)
}

#dictionay to store the cross validation results
cv_scores={}
for model_name,model in models.items():
  print(f"Training {model_name} with default parameters")
  cv_score=cross_val_score(model,X_train_smote,y_train_smote,cv=5)
  cv_scores[model_name]=cv_score
  print(f"{model_name} cross_validation accuracy :{np.mean(cv_score):.2f}")

cv_scores

"""Random forest gives the highest accuracy compared to the other models with default parameters"""

rfc=RandomForestClassifier(random_state=42)
rfc.fit(X_train_smote,y_train_smote)
print(y_test.value_counts())

"""Model evaluation"""

y_test_pred=rfc.predict(X_test)
print("Accuracy Score\n", accuracy_score(y_test,y_test_pred))
print("Confusion Matrix\n",confusion_matrix(y_test,y_test_pred))
print("Classification Report\n",classification_report(y_test,y_test_pred))

model_data={"model":rfc,"features_names":X.columns.tolist()}
# Save the trained model as a pickle file
with open("customerchurn_model.pkl","wb") as f:
  pickle.dump(model_data,f)

with open("/content/drive/MyDrive/customerchurn_model.pkl","rb") as f:
  model_data=pickle.load(f)
  loaded_model=model_data["model"]
  features_names=model_data["features_names"]

print(features_names)

input_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}
input_data_df=pd.DataFrame([input_data])

"""Load the saved model and build a Predictive System"""

print(input_data)

with open("/content/drive/MyDrive/encoders.pkl","rb") as f:
  encoders=pickle.load(f)

# encode categorical features using the saved encoders
for column,encoder in encoders.items():
  input_data_df[column]=encoder.transform(input_data_df[column])

from google.colab import drive
drive.mount('/content/drive')

print(input_data_df)

"""**Prediction**"""

predicted_churn=loaded_model.predict(input_data_df)
print(predicted_churn)

pred_prob=loaded_model.predict_proba(input_data_df)
print(pred_prob)

print(f"Prediction : {'Churn' if predicted_churn[0]==1 else 'No Churn'}")
print(f"Probability of Churn : {pred_prob[0][1]}")

