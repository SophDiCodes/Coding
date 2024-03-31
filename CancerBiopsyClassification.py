import gdown 
import pandas as pd
from sklearn import metrics

# gdown.download('https://drive.google.com/uc?id=1grV8hSxULsGvnbwEMPjPaknccfIOlcoB','cancer_data.csv',True);

from google.cloud import storage
def download_public_file(bucket_name, source_blob_name, destination_file_name):
    storage_client = storage.Client.create_anonymous_client()

    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print(
        "Downloaded public blob {} from bucket {} to {}.".format(
            source_blob_name, bucket.name, destination_file_name
        )
    )

download_public_file('inspirit-ai-data-bucket-1','Data/AI Scholars/Sessions 1 - 5/Session 2b - Logistic Regression/cancer.csv','cancer_data.csv')

data = pd.read_csv('cancer_data.csv')
data['diagnosis'].replace({'M':1, 'B':0}, inplace = True)
data.to_csv('cancer_data.csv')
del data

import os             
import numpy as np    
import pandas as pd   
from sklearn.metrics import accuracy_score   

data_path  = 'cancer_data.csv'

dataframe = pd.read_csv(data_path)

dataframe = dataframe[['diagnosis', 'perimeter_mean', 'radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'concavity_mean', 'symmetry_mean']]
dataframe['diagnosis_cat'] = dataframe['diagnosis'].astype('category').map({1: '1 (malignant)', 0: '0 (benign)'})

dataframe.head(5)

dataframe.info()

import seaborn as sns
import matplotlib.pyplot as plt 

sns.catplot(x = 'radius_mean', y = 'diagnosis_cat', data = dataframe, order=['1 (malignant)', '0 (benign)'])
dataframe.head()

from sklearn import linear_model

X,y = dataframe[['radius_mean']], dataframe[['diagnosis']]

model = linear_model.LinearRegression()
model.fit(X, y)
preds = model.predict(X)

sns.scatterplot(x='radius_mean', y='diagnosis', data=dataframe)
plt.plot(X, preds, color='r')
plt.legend(['Linear Regression Fit', 'Data'])

boundary = 15 

sns.catplot(x = 'radius_mean', y = 'diagnosis_cat', data = dataframe, order=['1 (malignant)', '0 (benign)'])
plt.plot([boundary, boundary], [-.2, 1.2], 'g', linewidth = 2)

def boundary_classifier(target_boundary, radius_mean_series):
  result = [] 
  for i in radius_mean_series:
    if i > target_boundary:
      result.append(1)
    else:
      result.append(0)
  return result

chosen_boundary = 15 

y_pred = boundary_classifier(chosen_boundary, dataframe['radius_mean'])
dataframe['predicted'] = y_pred

y_true = dataframe['diagnosis']

sns.catplot(x = 'radius_mean', y = 'diagnosis_cat', hue = 'predicted', data = dataframe, order=['1 (malignant)', '0 (benign)'])
plt.plot([chosen_boundary, chosen_boundary], [-.2, 1.2], 'g', linewidth = 2)

print (list(y_true))
print (y_pred)

accuracy = accuracy_score(y_true,y_pred)
print(accuracy)

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(dataframe, test_size = 0.2, random_state = 1)

print('Number of rows in training dataframe:', train_df.shape[0])
train_df.head()

print('Number of rows in test dataframe:', test_df.shape[0])
test_df.head()

X = ['radius_mean']
y = 'diagnosis'

X_train = train_df[X]
print('X_train, our input variables:')
print(X_train.head())
print()

y_train = train_df[y]
print('y_train, our output variable:')
print(y_train.head())

logreg_model = linear_model.LogisticRegression()

logreg_model.fit(X_train, y_train)

X_test = test_df[X]
y_test = test_df[y]

y_pred = logreg_model.predict(X_test)

test_df['predicted'] = y_pred.squeeze()
sns.catplot(x = X[0], y = 'diagnosis_cat', hue = 'predicted', data=test_df, order=['1 (malignant)', '0 (benign)'])

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)


y_prob = logreg_model.predict_proba(X_test)
X_test_view = X_test[X].values.squeeze()
plt.xlabel('radius_mean')
plt.ylabel('Predicted Probability')
sns.scatterplot(x = X_test_view, y = y_prob[:,1], hue = y_test, palette=['purple','green'])

dataframe.head(1)

X = ['area_mean']
y = 'diagnosis'

train_df, test_df = train_test_split(dataframe, test_size = 0.2, random_state = 1)

X_train = train_df[X]
X_test = test_df[X]

y_train = train_df[y]
y_test = test_df[y]

model = linear_model.LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy_score(y_test, y_pred)

dataframe.head(1)

ulti_X = ['area_mean', 'radius_mean', 'perimeter_mean', 'texture_mean', 'smoothness_mean', 'concavity_mean'] #Try changing this later!
y = 'diagnosis'

train_df, test_df = train_test_split(dataframe, test_size = 0.2, random_state = 1)

X_train = train_df[multi_X]
X_test = test_df[multi_X]
y_train = train_df[y]
y_test = test_df[y]
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy_score(y_test, y_pred)


# Import the metrics class
from sklearn import metrics

# Create the Confusion Matrix
# y_test = dataframe['diagnosis']
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

# Visualizing the Confusion Matrix
class_names = [0,1]

fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names)) 
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g') # Creating heatmap
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y = 1.1)
plt.ylabel('Actual diagnosis')
plt.xlabel('Predicted diagnosis')

print (cnf_matrix)
(tn, fp), (fn, tp) = cnf_matrix
print ("TN, FP, FN, TP:", tn, fp, fn, tp)

TPR = (tp)/(tp+fn)
print(TPR)

PPV = (tp)/(tp+fp)

print(TPR)
print(PPV)

from sklearn import tree

class_dt = tree.DecisionTreeClassifier(max_depth=3)

# We use our previous `X_train` and `y_train` sets to build the model
class_dt.fit(X_train, y_train)

#visualize and interpret tree
tree.plot_tree(class_dt) 

#predictions based on model
multi_y_pred = class_dt.predict(X_test)

#model performance
print("Accuracy: ", metrics.accuracy_score(y_test, multi_y_pred))
print("Precision: ", metrics.precision_score(y_test, multi_y_pred))
print("Recall: ", metrics.recall_score(y_test, multi_y_pred))



