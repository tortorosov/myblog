---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="_X9GuXoSXleA" -->
<center>
    <p style="text-align:center">
        <img alt="phoenix logo" src="https://storage.googleapis.com/arize-phoenix-assets/assets/phoenix-logo-light.svg" width="200"/>
        <br>
        <a href="https://docs.arize.com/phoenix/">Docs</a>
        |
        <a href="https://github.com/Arize-ai/phoenix">GitHub</a>
        |
        <a href="https://join.slack.com/t/arize-ai/shared_invite/zt-1px8dcmlf-fmThhDFD_V_48oU7ALan4Q">Community</a>
    </p>
</center>
<h1 align="center">Active Learning for a Drifting Image Classification Model</h1>

Imagine you're in charge of maintaining a model that classifies the action of people in photographs. Your model initially performs well in production, but its performance gradually degrades over time.

Phoenix helps you surface the reason for this regression by analyzing the embeddings representing each image. Your model was trained on crisp and high-resolution images, but as you'll discover, it's encountering blurred and noisy images in production that it can't correctly classify.

In this tutorial, you will:

- Download curated datasets of embeddings and predictions
- Define a schema to describe the format of your data
- Launch Phoenix to visually explore your embeddings
- Investigate problematic clusters
- Export problematic production data for labeling and fine-tuning

Let's get started!

## Install Dependencies and Import Libraries

Install Phoenix.
<!-- #endregion -->

```python jupyter={"source_hidden": true}
%pip install -Uq "arize-phoenix[embeddings]"
```

Import libraries.

```python jupyter={"source_hidden": true}
import pandas as pd
from IPython.display import HTML, display

import phoenix as px
```

<!-- #region id="OFeF5_Bysd2f" -->
## Download and Inspect the Data

Download production and training image data containing photographs of people performing various actions (sleeping, eating, running, etc.).
<!-- #endregion -->

```python jupyter={"source_hidden": true}
train_df = pd.read_parquet(
    "http://storage.googleapis.com/arize-phoenix-assets/datasets/unstructured/cv/human-actions/human_actions_training.parquet"
)
prod_df = pd.read_parquet(
    "http://storage.googleapis.com/arize-phoenix-assets/datasets/unstructured/cv/human-actions/human_actions_production.parquet"
)
```

View a few training data points.

```python jupyter={"source_hidden": true}
train_df.head()
```

The columns of the dataframe are:
- **prediction_id:** a unique identifier for each data point
- **prediction_ts:** the Unix timestamps of your predictions
- **url:** a link to the image data
- **image_vector:** the embedding vectors representing each image
- **actual_action:** the ground truth for each image
- **predicted_action:** the predicted class for the image

View a few production data points.

```python jupyter={"source_hidden": true}
prod_df.head()
```

Notice that the production data is missing ground truth, i.e., has no "actual_action" column.

Display a few images alongside their predicted and actual labels. 

```python jupyter={"source_hidden": true}
def display_examples(df):
    """
    Displays each image alongside the actual and predicted classes.
    """
    sample_df = df.reindex(columns=["actual_action", "predicted_action", "url"]).rename(
        columns={"url": "image"}
    )
    html = sample_df.to_html(
        escape=False, index=False, formatters={"image": lambda url: f'<img src="{url}">'}
    )
    display(HTML(html))


display_examples(train_df.head())
```

## Launch Phoenix

Define a schema to tell Phoenix what the columns of your training dataframe represent (features, predictions, actuals, tags, embeddings, etc.). See the [docs](https://docs.arize.com/phoenix/) for guides on how to define your own schema and API reference on `phoenix.Schema` and `phoenix.EmbeddingColumnNames`.

```python jupyter={"source_hidden": true}
train_schema = px.Schema(
    timestamp_column_name="prediction_ts",
    prediction_label_column_name="predicted_action",
    actual_label_column_name="actual_action",
    embedding_feature_column_names={
        "image_embedding": px.EmbeddingColumnNames(
            vector_column_name="image_vector",
            link_to_data_column_name="url",
        ),
    },
)
```

The schema for your production data is the same, except it does not have an actual label column.

```python jupyter={"source_hidden": true}
prod_schema = px.Schema(
    timestamp_column_name="prediction_ts",
    prediction_label_column_name="predicted_action",
    embedding_feature_column_names={
        "image_embedding": px.EmbeddingColumnNames(
            vector_column_name="image_vector",
            link_to_data_column_name="url",
        ),
    },
)
```

Create Phoenix datasets that wrap your dataframes with schemas that describe them.

```python jupyter={"source_hidden": true}
prod_ds = px.Inferences(dataframe=prod_df, schema=prod_schema, name="production")
train_ds = px.Inferences(dataframe=train_df, schema=train_schema, name="training")
```

Launch Phoenix. Follow the instructions in the UI to open the Phoenix UI.

```python jupyter={"source_hidden": true}
(session := px.launch_app(primary=prod_ds, reference=train_ds)).view()
```

## Find and Export Problematic Clusters

Click on "image_embedding" in the "Embeddings" section.

![click on image embedding](http://storage.googleapis.com/arize-phoenix-assets/assets/docs/notebooks/image-classification/click_on_image_embedding.png)

Select a period of high drift in the Euclidean distance graph at the top.

![select period of high drift](http://storage.googleapis.com/arize-phoenix-assets/assets/docs/notebooks/image-classification/select_period_of_high_drift.png)

Click on the top cluster in the panel on the left. Phoenix has identified this cluster as problematic because it consists entirely or almost entirely of production data, meaning that your model is making production inferences on data the likes of which it never saw during training.

![select top cluster](http://storage.googleapis.com/arize-phoenix-assets/assets/docs/notebooks/image-classification/select_top_cluster.png)

Use the panel at the bottom to examine the data points in this cluster. What do you notice about these data points that is different from the training data points you saw earlier?

![inspect points in cluster](http://storage.googleapis.com/arize-phoenix-assets/assets/docs/notebooks/image-classification/inspect_points_in_cluster.png)

The data points in the cluster above are grainy and noisy. Click on the "Export" button to save your cluster for relabeling and fine-tuning.

![export cluster](http://storage.googleapis.com/arize-phoenix-assets/assets/docs/notebooks/image-classification/export_cluster.png)

## Load and View Exported Data

View the exported cluster as a dataframe in your notebook.

```python jupyter={"source_hidden": true}
export_df = session.exports[-1]
export_df.head()
```

Display a few examples from your exported data.

```python jupyter={"source_hidden": true}
display_examples(export_df.head())
```

Congrats! You've pinpointed the blurry or noisy images that are hurting your model's performance in production. As an actionable next step, you can label your exported production data and fine-tune your model to improve performance.
