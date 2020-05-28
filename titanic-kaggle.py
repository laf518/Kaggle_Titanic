from pandas import read_csv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Reading the csv data
data = read_csv("train.csv");

#Spilting Name column to name & title 
temp1 = data["Name"].str.split(" ", n=1, expand = True);
temp2  = temp1[1].str.split(".", n=1, expand = True);
data["title"] = temp2[0];
data["Name"] = temp1[0] + " " + temp2[1];

#Replace missing values with mean
data = data.fillna(data.mean());

# One hot encoding for categorical features into numeric
categorical_feature_mask = data.dtypes==object;
categorical_cols = data.columns[categorical_feature_mask].tolist();
labee = LabelEncoder();
data[categorical_cols] = data[categorical_cols].apply(lambda col: labee.fit_transform(col));

print(categorical_cols);
"""
#Separating features & target
temp_data = data.copy();
del temp_data["Survived"];
feature_data = temp_data;
target_data = data["Survived"];

#print(feature_data["title"]);


#Split data into train & test to evaluate based on generalization performance
x_train, x_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.25, random_state=0);

model = LogisticRegression().fit(x_train, y_train);
"""


