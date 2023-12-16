import os
import modal
from PIL import Image
    
LOCAL=False
FG_VERSION = 2
FV_VERSION = 2
MODEL_VERSION = 7
PRE_FG_VERSION = 2

if LOCAL == False:
   stub = modal.Stub("wine_batch_inference")
   hopsworks_image = modal.Image.debian_slim(python_version='3.9').pip_install(["hopsworks","joblib","seaborn","scikit-learn==1.1.1","dataframe_image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns
    import requests  

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=MODEL_VERSION)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    feature_view = fs.get_feature_view(name="wine", version=FV_VERSION)
    batch_data = feature_view.get_batch_data()
    
    # only need to get prediction for the latest added wine, use offset to get the last wine
    y_pred = model.predict(batch_data)
    offset = 1
    quality = y_pred[y_pred.size - offset]
    print("Quality predicted: " + str(quality))
    dataset_api = project.get_dataset_api()
   
    wine_fg = fs.get_feature_group(name="wine", version=FG_VERSION)
    df = wine_fg.read() 
    #print(df)
    label = df.iloc[-offset]["quality"]
    
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=PRE_FG_VERSION,
                                                primary_key=["datetime"],
                                                description="Wine Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [quality],
        'label': [int(label)],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read(read_options={"use_hive": True})
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    df_recent = history_df.tail(4)
    dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]

    # Only create the confusion matrix when our wine_predictions feature group has examples of at least 2 wine qualities
    print("Number of different wine quality predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() > 1:
        results = confusion_matrix(labels, predictions)
        diff_quality = pd.concat([predictions['prediction'], labels['label']]).unique()
        diff_quality.sort()
        true_quality = []
        pred_quality = []
        for i in diff_quality:
            true_quality.append('True quality' + str(i))
            pred_quality.append('Pred quality' + str(i))
        df_cm = pd.DataFrame(results, true_quality, pred_quality)
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need at least 2 different quality predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different wine quality predictions") 


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        modal.runner.deploy_stub(stub)
        with stub.run():
            f.remote()

