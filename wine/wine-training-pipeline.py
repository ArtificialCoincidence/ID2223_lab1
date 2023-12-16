# %%
import hopsworks
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
from matplotlib import pyplot
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import joblib
import os

# %%
import os

FG_VERSION = 2 # wine feature group version
FV_VERSION = 2 # wine feature view version
 
for k, v in os.environ.items():
    if "HOPS" in k:
        print(f'{k}={v}')

# %%
# You have to set the environment variable 'HOPSWORKS_API_KEY' for login to succeed
project = hopsworks.login()
fs = project.get_feature_store()

# %%
# The feature view is the input set of features for your model. The features can come from different feature groups.    
# You can select features from different feature groups and join them together to create a feature view
wine_fg = fs.get_feature_group(name="wine", version=FG_VERSION)
query = wine_fg.select_all()
feature_view = fs.get_or_create_feature_view(name="wine",
                                  version=FV_VERSION,
                                  description="Read from wine dataset",
                                  labels=["quality"],
                                  query=query)

# %%
# You can read training data, randomly split into train/test sets of features (X) and labels (y)        
X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)

# Train our model with the logistic regression algorithm of sklearn using our features (X_train) and labels (y_train)
model = linear_model.LogisticRegression()
model.fit(X_train, y_train)

# %%
# Evaluate model performance using the features from the test set (X_test)
y_pred = model.predict(X_test)

# Compare predictions (y_pred) with the labels in the test set (y_test)
metrics = classification_report(y_test, y_pred, output_dict=True)
results = confusion_matrix(y_test, y_pred)

# %%
# Create the confusion matrix as a figure, we will later store it as a PNG image file
diff_quality = np.unique(np.concatenate([y_pred, y_test], axis=None)) # get unique values in matrix
print(diff_quality)
true_quality = []
pred_quality = []
for i in diff_quality:
    true_quality.append('True quality' + str(i))
    pred_quality.append('Pred quality' + str(i))
df_cm = pd.DataFrame(results, true_quality, pred_quality)
cm = sns.heatmap(df_cm, annot=True)
fig = cm.get_figure()

# %%
# We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
mr = project.get_model_registry()

# The contents of the 'wine_model' directory will be saved to the model registry. Create the dir, first.
model_dir="wine_model"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)

# Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
joblib.dump(model, model_dir + "/wine_model.pkl")
fig.savefig(model_dir + "/confusion_matrix.png")    

# Specify the schema of the model's input/output using the features (X_train) and labels (y_train)
input_schema = Schema(X_train)
output_schema = Schema(y_train)
model_schema = ModelSchema(input_schema, output_schema)

# Create an entry in the model registry that includes the model's name, desc, metrics
wine_model = mr.python.create_model(
    name="wine_model", 
    metrics={"accuracy" : metrics['accuracy']},
    model_schema=model_schema,
    description="Wine Predictor"
)

# Upload the model to the model registry, including all files in 'model_dir'
wine_model.save(model_dir)

# %%



