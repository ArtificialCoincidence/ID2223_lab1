# %%
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
#matplotlib inline

# %%
import hopsworks
import pandas as pd

# %%
project = hopsworks.login()
fs = project.get_feature_store()

# %%
iris_df = pd.read_csv("https://repo.hops.works/master/hopsworks-tutorials/data/iris.csv")
iris_df

# %%
iris_df.info()

# %%
iris_df.describe()

# %%
iris_df['variety'].value_counts()

# %% [markdown]
# ### Exploratory Data Analysis (EDA)  our Iris Data
# 
# Let's look at our iris flower - the distribution and range of values for the 4 different features
#  * sepal_length
#  * sepal_width
#  * petal_length
#  * petal_width
#  
#  and the target variable is `variety`.

# %%
g = sns.pairplot(iris_df, hue='variety', markers='+')
plt.show()

# %% [markdown]
# ### Visualize range of values 
# 
# We want to design a simulator generate the 3 types of iris flower (setosa, versicolor, virginica).
# To do this, we can visualize the range of values for the length and width of the sepal and petal for each of the 3 flowers.

# %%
g = sns.violinplot(y='variety', x='sepal_length', data=iris_df, inner='quartile')
plt.show()
g = sns.violinplot(y='variety', x='sepal_width', data=iris_df, inner='quartile')
plt.show()
g = sns.violinplot(y='variety', x='petal_length', data=iris_df, inner='quartile')
plt.show()
g = sns.violinplot(y='variety', x='petal_width', data=iris_df, inner='quartile')
plt.show()

# %% [markdown]
# ### Insert our Iris DataFrame into a FeatureGroup
# Let's write our historical iris feature values and labels to a feature group.
# When you write historical data, this process is called `backfilling`.

# %%
iris_fg = fs.get_or_create_feature_group(
    name="iris",
    version=2,
    primary_key=["sepal_length","sepal_width","petal_length","petal_width"], 
    description="Iris flower dataset")
iris_fg.insert(iris_df)

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
suite = ExpectationSuite(expectation_suite_name="iris_dimensions")

expect(suite, "sepal_length", 4.5, 8.0)
expect(suite, "sepal_width", 2.1, 4.5)
expect(suite, "petal_length", 1.2, 7)
expect(suite, "petal_width", 0.2, 2.5)
iris_fg.save_expectation_suite(expectation_suite=suite, validation_ingestion_policy="STRICT")    

# %%



