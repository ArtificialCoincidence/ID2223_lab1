import os
import modal

LOCAL=False
FG_VERSION = 2

if LOCAL == False:
   stub = modal.Stub("wine_daily")
   image = modal.Image.debian_slim().pip_install(["hopsworks"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"),
                  mounts=[modal.Mount.from_local_dir("/Users/dairuijia/documents/ScalableMachineLearning/lab1/code/wine/serverless-ml/wine_configuration", remote_path="/root/wine_configuration")])
   def f():
       g()

def generate_wine():
    """
    Returns a single wine as a single row in a DataFrame
    """
    import pandas as pd
    import random

    if LOCAL==True:
        wine_parameters = pd.read_csv('./wine_parameters.csv')
        wine_parameter_min = pd.read_csv('./wine_parameter_min.csv')
        wine_parameter_max = pd.read_csv('./wine_parameter_max.csv')
    else:
        wine_parameters = pd.read_csv('/root/wine_configuration/wine_parameters.csv')
        wine_parameter_min = pd.read_csv('/root/wine_configuration/wine_parameter_min.csv')
        wine_parameter_max = pd.read_csv('/root/wine_configuration/wine_parameter_max.csv')
    print(wine_parameter_max)

    wine_df = pd.DataFrame(columns=wine_parameters["parameter"].transpose(), dtype='float64')

    for i in range(11):
        parameter = wine_parameters._get_value(i, 'parameter')
        min_value = wine_parameter_max._get_value(i, 'wine_parameter_max')
        max_value = wine_parameter_min._get_value(i, 'wine_parameter_min')
        wine_df.at[0, parameter] = random.uniform(min_value, max_value)
    wine_df.at[0, 'color'] = random.randint(0, 1)
    wine_df.at[0, 'quality'] = random.randint(0, 12)
    wine_df['color'] = wine_df['color'].astype('int64')
    wine_df['quality'] = wine_df['quality'].astype('int64')
    print(wine_df)
    return wine_df


def get_random_wine():
    """
    Returns a DataFrame containing one random wine
    """
    import pandas as pd
    import random

    wine_df = generate_wine()

    # write it to the featurestore
    return wine_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_df = get_random_wine()

    wine_fg = fs.get_feature_group(name="wine",version=FG_VERSION)
    wine_fg.insert(wine_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        modal.runner.deploy_stub(stub)
        with stub.run():
            f.remote()
