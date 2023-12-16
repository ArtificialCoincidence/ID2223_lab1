# %%
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
#matplotlib inline

# %%
import hopsworks
import pandas as pd

API_KEY = 'IjpOAHZvfhOIKz5Z.nSNpsCdomwyvjmYZ8utAUd1HE8up18q1eoi4oA41wMMHJlSbVK8mZNUJPbOjs1li' # hopsworks api key value
FG_VERSION = 2 # wine feature group version

# %%
project = hopsworks.login(api_key_value=API_KEY)
fs = project.get_feature_store()

# %%
wine_cols = ['fixed_acidity', 'volatile_acidity', 'citric_aicd', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide',
                'total_sulfur_dioxide', 'density', 'ph', 'sulphates', 'alcohol', 'quality']
redwine_df = pd.read_table("./wine+quality/winequality-red.csv", sep=';', header=None, names=wine_cols, skiprows=1) # red wine dataset
redwine_df.insert(11, 'color', 0) # 0 represent red wine

whitewine_df = pd.read_table("./wine+quality/winequality-white.csv", sep=';', header=None, names=wine_cols, skiprows=1) # white wine dataset
whitewine_df.insert(11, 'color', 1) # 1 represent white wine

wine_df = pd.concat([redwine_df, whitewine_df], axis=0, ignore_index=True) # integrate two datasets to one
wine_df.head()
wine_df.to_csv('./wine_dataset.csv', index=False) # save clean wine dataframe as file for further reuse
# %%
wine_df.info()

# %%
wine_df.describe()

# %%
wine_df['quality'].value_counts()

# %% [markdown]
# ### Exploratory Data Analysis (EDA)  our Wine Data
# 
# Let's look at our wine - the distribution and range of values for the different qualities
# the target variable is `quality`

# %%
g = sns.pairplot(wine_df, hue='quality', markers='+')
plt.show()

# %% [markdown]
# ### Visualize range of values 
# 
# We want to design a simulator generate a new wine.
# To do this, we can visualize the range of values for each parameter.

# %%
wine_parameters = wine_cols[:11] + ['color'] + ['quality']
wine_parameters_df = pd.DataFrame(wine_parameters)
wine_parameters_df.to_csv('./wine_configuration/wine_parameters.csv', index=None, header=['parameter'])
wine_parameters_df.head()
for parameter in wine_parameters:
    g = sns.violinplot(y='quality', x=parameter, data=wine_df, inner='quartile')
    plt.show()

# %% [markdown]
# ### Insert our Wine DataFrame into a FeatureGroup
# Let's write our historical wine feature values and labels to a feature group.
# When you write historical data, this process is called `backfilling`.

# %%
wine_fg = fs.get_or_create_feature_group(
    name="wine",
    version=FG_VERSION,
    primary_key=wine_parameters, 
    description="Wine dataset")
wine_fg.insert(wine_df)
#??

# %% [markdown]
# ### Data Validation
# If you want, you can enable data validation for your feature group.
# The code below will prevent iris flower data from being written your your feature group if you write values outside the expected ranges.

# %%
from great_expectations.core import ExpectationSuite, ExpectationConfiguration

def expect(suite, column, min_val, max_val):
    suite.add_expectation(
    ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={
            "column":column, 
            "min_value":min_val,
            "max_value":max_val,
        }
    )
)

# %%
suite = ExpectationSuite(expectation_suite_name="wine_dimensions")

wine_parameter_min = wine_df.min(axis=0)
wine_parameter_max = wine_df.max(axis=0)
# save min and max value of each parameter for future reuse
wine_parameter_min[:11].to_csv('./wine_configuration/wine_parameter_min.csv', header=['wine_parameter_min'])
wine_parameter_max[:11].to_csv('./wine_configuration/wine_parameter_max.csv', header=['wine_parameter_max'])
wine_parameter_min.head()
i = 0
for parameter in wine_parameters[:11]:
    expect(suite, parameter, wine_parameter_min[i], wine_parameter_max[i])
    i += 1
expect(suite, 'color', 0, 1)
expect(suite, 'quality', 0, 10)
wine_fg.save_expectation_suite(expectation_suite=suite, validation_ingestion_policy="STRICT", overwrite=True)    

# %%



