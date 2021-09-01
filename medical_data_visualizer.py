import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv("medical_examination.csv")

# Add 'overweight' column
df["overweight"] = df["weight"] / ((df["height"] / 100) * (df["height"] / 100))
df.loc[df["overweight"] <= 25, "overweight"] = 0
df.loc[df["overweight"] > 25, "overweight"] = 1

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df.loc[df["cholesterol"] == 1, "cholesterol"] = 0
df.loc[df["cholesterol"] > 1, "cholesterol"] = 1

df.loc[df["gluc"] == 1, "gluc"] = 0
df.loc[df["gluc"] > 1, "gluc"] = 1


# Draw Categorical Plot
def draw_cat_plot():
    """
    Plots the categories of the dataset

    Returns:
        Figure: Returns the plot that was created
    """
    # Creates a new Dataframe with 'cardio' acting as the ID
    df_cat = pd.melt(df, id_vars=["cardio"],
                     value_vars=["cholesterol", "gluc", "smoke", "alco", "active", "overweight"])

    df_cat = df_cat.sort_values(by="variable")
    df_cat["value"] = df_cat["value"].astype(int)

    # Groups and reformats data by 'cardio' and adds 'total'
    df_cat["total"] = None
    df_cat = df_cat.groupby(["cardio", "value", "variable"])
    df_cat = df_cat.agg(["size"])
    df_cat = df_cat.reset_index()
    df_cat.columns = ["__".join(col).strip() for col in df_cat.columns.values]
    df_cat = df_cat.rename(columns={
                           "cardio__": "cardio", "value__": "value", "variable__": "variable", "total__size": "total"})

    # Draw the catplot with 'sns.catplot()'
    graph = sns.catplot(y="total", x='variable', hue="value",
                        kind="bar", col='cardio', data=df_cat)
    fig = graph.fig

    # Do not modify the next two lines
    fig.savefig("catplot.png")
    return fig


# Draw Heat Map
def draw_heat_map():
    """
    Creates a heatmap from the dataset

    Returns:
        Figure: Returns the heatmap that was created
    """
    # Clean data
    df_heat = df[df["ap_lo"] <= df["ap_hi"]]
    df_heat = df[(df["height"] >= df["height"].quantile(0.025)) & (df["height"] <= df["height"].quantile(
        0.975)) & (df["weight"] >= df["weight"].quantile(0.025)) & (df["weight"] <= df["weight"].quantile(0.975))]

    # Finds the correlation matrix
    corr = df_heat.corr()

    # Generates a mask
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # Draw the heatmap with 'sns.heatmap()'
    ax = sns.heatmap(corr, linewidths=.5, annot=True, fmt=".1f", mask=mask, square=True,
                     center=0, vmin=0.1, vmax=0.25, cbar_kws={"shrink": .45, "format": "%.2f"})

    # Do not modify the next two lines
    fig.savefig("heatmap.png")
    return fig
