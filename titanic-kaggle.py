import pandas as pd
from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import accuracy_score

#Reading the csv data
data = read_csv("train.csv");

#Spliting Name column to name & title 
temp1 = data["Name"].str.split(" ", n=1, expand = True);
temp2  = temp1[1].str.split(".", n=1, expand = True);
data["title"] = temp2[0];
data["Name"] = temp1[0] + " " + temp2[1];
# del data["Name"]
# del data["title"]
# del data["Ticket"]

#Create binary values for Gender
data["Male"] = data["Sex"].replace({"male": 1, "female": 0})
data["Female"] = data["Sex"].replace({"male": 0, "female": 1})
del data["Sex"]

# Creating deck level feature
data["Deck_Letter"] = data["Cabin"].str.get(0)
data["Deck"] = pd.to_numeric(data["Deck_Letter"].replace({"T": 0, "A": 1, "B": 2,
    "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}))
deck_avg = data["Deck"].mean()
data["Deck"] = data["Deck"].fillna(deck_avg)
del data["Deck_Letter"]
del data["Cabin"]


# Label encoding for features missing values
labee = LabelEncoder();
data = data.apply(lambda col: labee.fit_transform(col.astype(str)), axis=0, result_type='expand');
#data.to_excel('titanic.xlsx', sheet_name='training')
#Checking if any missing vals, below gives empty dataframe meaning no missing vals
#print(data[data.isnull().any(axis=1)]);

#Separating features & target
temp_data = data.copy();
del temp_data["Survived"];
feature_data = temp_data;
target_data = data["Survived"];

#Checking if features & targets are separated correctly
#print(feature_data);
#print(target_data);

#Split data into train & test to evaluate based on generalization performance
x_train, x_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.25, random_state=0);

#Checking if spilt happened
#print(len(x_train));
#print(len(y_train));
#print(len(x_test));
#print(len(y_test));

#Scaling values
x_train = preprocessing.scale(x_train);

#print(x_train);

#Logistic Regression
lr = LogisticRegression();
model = lr.fit(x_train, y_train);

y_pred=model.predict(x_test);
cnf_matrix = metrics.confusion_matrix(y_test, y_pred);
print()
print('Logistic Regression:')
print('--------------')
print()
print('Confusion Matrix:')
print(cnf_matrix);
print()
print('Accuracy: ', accuracy_score(y_test, y_pred));

#Random Forest
rf = RandomForestClassifier(random_state=1)
model = rf.fit(x_train, y_train)

y_pred = model.predict(x_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print()
print('Random Forest:')
print('--------------')
print()
print('Confusion Matrix:')
print(cnf_matrix);
print()
print('Accuracy: ', accuracy_score(y_test, y_pred))