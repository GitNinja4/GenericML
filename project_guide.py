"""Plain-language guide to GenericML Studio.

WHAT THIS PROJECT DOES
======================

GenericML Studio is a Streamlit app for exploring a CSV dataset and preparing
classification or regression models. The usable workflow currently goes from
dataset upload through model training. Tuning, evaluation, MLflow tracking, and
prediction are shown in the navigation, but those pages are still placeholders.


HOW TO START THE APP
====================

1. Open a terminal in the project folder.
2. Install the packages with: pip install -r requirements.txt
3. Start the app with: streamlit run app.py
4. Open the local URL printed by Streamlit in your browser.

The first page asks for a CSV file. A small example CSV is kept in
data/sample/model_ui_check.csv.


THE FULL CURRENT WORKFLOW
========================

1. Upload and configure a dataset
---------------------------------

The Dataset page loads the CSV into a pandas DataFrame, checks that it has
rows and columns, and rejects duplicate column names. It shows a preview,
column types, missing-value counts, and other basic summaries. Choose the
column the model should predict (the target). The app suggests whether the
task is classification or regression; the user can change that choice.
Missing target values are reported, but the split step will not accept them;
remove or correct those rows before splitting.

2. Explore the data
-------------------

The EDA page summarizes the dataset and displays statistics and charts. It
helps identify missing values, duplicates, unusual values, feature types,
correlations, and target distributions. EDA is for inspection: it does not
apply preprocessing to the dataset.

3. Split the raw data
--------------------

The Train / Test Split page separates the target (y) from the input columns
(X), then creates four partitions:

    X_train, y_train  - used to configure and fit models
    X_test,  y_test   - held aside for a later, final evaluation stage

The default test share is 20 percent. The split has a random state so it can
be reproduced. Classification can use stratification to keep class shares
similar in both partitions. The split happens before any preprocessing is fit.

4. Configure preprocessing
--------------------------

The Preprocessing page lets the user choose which input features to keep and
how to handle their values. Options include filling missing numeric or
categorical values, clipping or removing numeric outliers, encoding categories,
scaling numeric values, and optional SelectKBest feature selection.

Applying the settings builds and saves an unfitted scikit-learn pipeline. It
does not fit that pipeline on the complete uploaded dataset. Fitting happens
later using training rows only. If the outlier setting is Remove, rows are
removed from training data only; the held-out test data stays unchanged.

5. Select models
----------------

The Models page reads available algorithms and their parameter controls from
src/models/registry.py. Models are grouped by problem type and family. The user
can select one or more compatible models and adjust the parameters exposed by
the registry. src/models/model_factory.py creates a configured estimator.

6. Train and validate models
----------------------------

The Training page requires a current split, an applied preprocessing setup,
and at least one selected model. It passes X_train and y_train to the training
workflow. It does not pass X_test or y_test.

For validation, classification uses Stratified K-Fold and regression uses
K-Fold. Each fold gets a fresh copy of the preprocessing and model pipeline,
so each fold learns transformations only from that fold's training rows. Once
After validation, a final pipeline is fitted on the training partition. If
training-only outlier removal is enabled, rows it removes are excluded from
this final fit.

Optional SMOTE or SMOTENC oversampling is available for classification. It is
disabled by default and runs inside the training pipeline, never on validation
or held-out test rows. The page reports training metrics, cross-validation
scores when enabled, timing, warnings, and a basic possible overfit/underfit
diagnostic. These scores are development results, not final test-set results.

7. Later stages
---------------

Tuning, Evaluation, MLflow, and Prediction are listed in the sidebar, but
currently render placeholder pages. The intended next use of the holdout is
to evaluate a chosen trained pipeline once. Do not treat the current training
scores as a substitute for that final evaluation.


HOW THE APP IS PUT TOGETHER
===========================

app.py is the entry point. It configures Streamlit, initializes session state,
loads shared styling and the sidebar, then renders the page selected in the
sidebar. ui/components.py contains shared layout, navigation, and progress
helpers. ui/sidebar.py draws the workflow navigation.

config/
    Application names and default values such as split size and random seed.

data/sample/
    A small CSV that can be used to try the interface.

src/data/
    CSV loading, data validation, target checks, dataset summaries, and EDA
    calculations.

src/preprocessing/
    Building blocks for missing-value handling, outlier handling, categorical
    encoding, scaling, feature selection, and assembled preprocessing pipes.

src/training/
    Raw train/test splitting, model pipeline assembly, optional class
    oversampling, cross-validation helpers, and the main training workflow.

src/models/
    The model registry, model factory, and classification/regression model
    helpers. Add or adjust registered model choices here.

src/evaluation/, src/tuning/, src/mlflow/, src/prediction/
    Folders reserved for later workflow stages. Some metric or support modules
    are present, but their pages are not currently connected as usable stages.

src/visualization/
    Plot helpers used by EDA, model evaluation, and feature-importance views.

src/utils/
    Shared exceptions, logging, small helpers, and Streamlit session-state
    initialization/reset logic.

ui/tabs/
    The user-facing pages. dataset.py, eda.py, train_test_split.py,
    preprocessing.py, models.py, and training.py implement the current flow.
    placeholders.py renders the later pages that are not ready yet.

artifacts/
    Output/artifact location used by the project. Do not remove it if the app
    or a future workflow is writing files there.


IMPORTANT DATA AND SESSION RULES
===============================

The uploaded dataset, choices, split partitions, and fitted models are kept in
Streamlit session state. They are temporary to the running app session; this
project does not currently promise that a model or dataset will still be
available after the session ends.

Changing the uploaded dataset, target, problem type, split settings, or
preprocessing choices invalidates dependent results. Recreate the split and
rerun later steps after changing an earlier choice.

The central safety rule is: split first, then fit preprocessing, sampling,
validation folds, and models using training data only. Keep the test partition
untouched until the future Evaluation stage.


PROJECT STATUS
==============

The test/ directory and README.md were removed from the repository during
project cleanup. There is currently no checked-in automated test suite, even
though pytest remains listed in requirements.txt. This guide is documentation
stored as a Python module; it is not imported by the app and does not change
application behavior.
"""