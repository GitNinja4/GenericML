import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, make_classification, make_regression
from streamlit.testing.v1 import AppTest
from pathlib import Path

from src.models.registry import get_available_models, get_model_families
from src.preprocessing.pipeline_builder import PreprocessingConfig, build_preprocessing_pipeline
from src.training.workflow import split_dataset


def _seed_app(dataset: pd.DataFrame, target: str, problem_type: str) -> AppTest:
    feature_columns = [column for column in dataset.columns if column != target]
    config = PreprocessingConfig(selected_features=feature_columns, target_column=target, problem_type=problem_type)
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=20)
    app.session_state["dataset"] = dataset
    app.session_state["dataset_valid"] = True
    app.session_state["dataset_name"] = "audit.csv"
    app.session_state["target_column"] = target
    app.session_state["problem_type"] = problem_type
    app.session_state["detected_problem_type"] = problem_type
    app.session_state["current_page"] = "preprocessing"
    app.session_state["target_n_classes"] = int(dataset[target].nunique())
    app.session_state["prep_selected_features"] = feature_columns
    app.session_state["selected_models"] = ["Logistic Regression"] if problem_type == "classification" else ["Ridge"]
    return app


def _apply_and_train(app: AppTest) -> None:
    app.run()
    assert not app.exception, app.exception
    apply_index = [button.label for button in app.button].index("Apply preprocessing configuration")
    app.button[apply_index].click().run()
    assert not app.exception, app.exception
    models_index = [button.label for button in app.button].index("Configure models →")
    app.button[models_index].click().run()
    assert not app.exception, app.exception
    train_index = [button.label for button in app.button].index("Train selected models")
    app.button[train_index].click().run()
    assert not app.exception, app.exception


def test_fresh_session_has_useful_empty_states() -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15).run()
    assert not app.exception
    assert "Please upload a CSV dataset to continue." in [item.value for item in app.info]
    assert any("Dataset" in button.label for button in app.button)
    assert not app.success


def test_app_uses_genericml_studio_branding() -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15).run()
    assert not app.exception
    rendered = "\n".join(item.value for item in app.markdown)
    assert "GenericML Studio" in rendered
    assert "MLFlow Studio" not in rendered


def test_file_upload_initializes_dataset_state() -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15).run()
    app.file_uploader[0].upload(
        "audit.csv",
        b"age,kind,target\n20,A,0\n30,B,1\n40,A,0\n50,B,1\n",
        "text/csv",
    ).run()
    assert not app.exception, app.exception
    assert app.session_state["dataset_name"] == "audit.csv"
    assert app.session_state["dataset_valid"] is True


@pytest.mark.parametrize(
    ("filename", "contents", "message"),
    [
        ("duplicate_headers.csv", b"value,value,target\n1,2,0\n3,4,1\n", "Duplicate column names detected"),
        ("empty.csv", b"", "empty"),
    ],
)
def test_invalid_csv_upload_shows_clear_error_without_app_exception(
    filename: str,
    contents: bytes,
    message: str,
) -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15).run()
    app.file_uploader[0].upload(filename, contents, "text/csv").run()

    assert not app.exception, app.exception
    assert any(message.casefold() in item.value.casefold() for item in app.error)
    assert app.session_state["dataset"] is None


def test_primary_ctas_navigate_between_active_pages() -> None:
    values, labels = make_classification(n_samples=30, n_features=3, n_informative=2, n_redundant=0, random_state=5)
    dataset = pd.DataFrame(values, columns=["a", "b", "c"])
    dataset["target"] = labels
    app = _seed_app(dataset, "target", "classification")
    app.session_state["current_page"] = "dataset"
    app.run()
    dataset_cta = next(button for button in app.button if "Continue to EDA" in button.label)
    dataset_cta.click().run()
    assert app.session_state["current_page"] == "eda"
    eda_cta = next(button for button in app.button if "Let's do preprocessing" in button.label)
    eda_cta.click().run()
    assert app.session_state["current_page"] == "preprocessing"


def test_new_file_upload_clears_downstream_state() -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15).run()
    app.file_uploader[0].upload(
        "first.csv",
        b"age,target\n20,0\n30,1\n40,0\n50,1\n",
        "text/csv",
    ).run()
    app.session_state["preprocessing_applied"] = True
    app.session_state["preprocessing_config"] = {"old": True}
    app.session_state["trained_models"] = {"old": {"status": "trained"}}
    app.file_uploader[0].upload(
        "second.csv",
        b"score,target\n1.0,10\n2.0,20\n3.0,30\n4.0,40\n",
        "text/csv",
    ).run()
    assert not app.exception, app.exception
    assert app.session_state["dataset_name"] == "second.csv"
    assert app.session_state["preprocessing_applied"] is False
    assert not app.session_state["preprocessing_config"]
    assert not app.session_state["trained_models"]


def test_same_filename_with_new_contents_clears_downstream_state() -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15).run()
    app.file_uploader[0].upload(
        "reused.csv",
        b"age,target\n20,0\n30,1\n40,0\n50,1\n",
        "text/csv",
    ).run()
    app.session_state["preprocessing_applied"] = True
    app.session_state["preprocessing_config"] = {"old_dataset": True}
    app.session_state["trained_models"] = {"old_model": {"status": "trained"}}
    app.session_state["training_results"] = {"old_model": {"status": "trained"}}

    app.file_uploader[0].upload(
        "reused.csv",
        b"age,target\n200,0\n300,1\n400,0\n500,1\n",
        "text/csv",
    ).run()

    assert not app.exception, app.exception
    assert app.session_state["dataset"].columns.tolist() == ["age", "target"]
    assert app.session_state["dataset"].iloc[0]["age"] == 200
    assert app.session_state["preprocessing_applied"] is False
    assert not app.session_state["preprocessing_config"]
    assert not app.session_state["trained_models"]
    assert not app.session_state["training_results"]


def test_preprocessing_defaults_to_all_non_target_features() -> None:
    dataset = pd.DataFrame({"value": [1.0, 2.0, 3.0], "group": ["A", "B", "A"], "target": [1.0, 2.0, 3.0]})
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15)
    for key, value in {
        "current_page": "preprocessing",
        "dataset": dataset,
        "dataset_name": "features.csv",
        "dataset_valid": True,
        "target_column": "target",
        "problem_type": "regression",
        "detected_problem_type": "regression",
        "prep_selected_features": [],
        "prep_features_initialized": False,
    }.items():
        app.session_state[key] = value

    app.run()

    assert not app.exception, app.exception
    assert app.multiselect(key="prep_selected_features").value == ["value", "group"]


def test_manual_problem_type_override_survives_page_navigation() -> None:
    dataset = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0], "target": [0.5, 1.5, 2.5, 3.5]})
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15)
    for key, value in {
        "current_page": "dataset",
        "dataset": dataset,
        "dataset_name": "override.csv",
        "dataset_valid": True,
        "target_column": "target",
        "problem_type": None,
        "detected_problem_type": None,
    }.items():
        app.session_state[key] = value

    app.run()
    assert not app.exception, app.exception
    assert app.session_state["problem_type"] == "regression"
    app.radio(key="problem_type_selector").set_value("Classification").run()
    assert not app.exception, app.exception
    assert app.session_state["problem_type"] == "classification"

    next(button for button in app.button if "EDA" in button.label).click().run()
    next(button for button in app.button if "Dataset" in button.label).click().run()
    assert not app.exception, app.exception
    assert app.radio(key="problem_type_selector").value == "Classification"
    assert app.session_state["problem_type"] == "classification"


def test_invalid_target_change_invalidates_previous_preprocessing_and_training() -> None:
    dataset = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0],
            "valid_target": [0, 1, 0, 1],
            "empty_target": [None, None, None, None],
        }
    )
    config = PreprocessingConfig(selected_features=["feature"], target_column="valid_target", problem_type="classification")
    pipeline = build_preprocessing_pipeline(config, ["feature"], [])
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15)
    for key, value in {
        "current_page": "dataset",
        "dataset": dataset,
        "dataset_name": "invalid-target.csv",
        "dataset_valid": True,
        "target_column": "valid_target",
        "problem_type": "classification",
        "detected_problem_type": "classification",
        "preprocessing_config": config.to_dict(),
        "preprocessing_pipeline": pipeline,
        "selected_features": ["feature"],
        "preprocessing_applied": True,
        "selected_models": ["Logistic Regression"],
        "trained_models": {"Logistic Regression": {"status": "trained"}},
    }.items():
        app.session_state[key] = value

    app.run()
    next(selectbox for selectbox in app.selectbox if selectbox.label == "Select target column").select("empty_target").run()

    assert "no usable values" in " ".join(item.value for item in app.error).casefold()
    assert app.session_state["target_column"] == "empty_target"
    assert app.session_state["preprocessing_applied"] is False
    assert app.session_state["preprocessing_pipeline"] is None
    assert not app.session_state["trained_models"]


def test_sidebar_marks_eda_complete_only_after_visit() -> None:
    dataset = pd.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0], "target": [0, 1, 0, 1]})
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=15)
    app.session_state["dataset"] = dataset
    app.session_state["dataset_valid"] = True
    app.session_state["target_column"] = "target"
    app.session_state["problem_type"] = "classification"
    app.run()

    eda_button = next(button for button in app.button if button.key == "nav_eda")
    assert eda_button.label.startswith("○")
    eda_button.click().run()
    assert not app.exception, app.exception
    assert next(button for button in app.button if button.key == "nav_eda").label.startswith("●")
    next(button for button in app.button if button.key == "nav_dataset").click().run()
    assert not app.exception, app.exception
    assert next(button for button in app.button if button.key == "nav_eda").label.startswith("✓")


def test_eda_rendering_does_not_mutate_raw_dataset() -> None:
    dataset = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "group": ["A", "B", "A", "B", "A", "B"],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )
    original = dataset.copy(deep=True)
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=20)
    for key, value in {
        "current_page": "eda",
        "dataset": dataset,
        "dataset_name": "eda-raw.csv",
        "dataset_valid": True,
        "target_column": "target",
        "problem_type": "classification",
        "detected_problem_type": "classification",
    }.items():
        app.session_state[key] = value

    app.run()

    assert not app.exception, app.exception
    pd.testing.assert_frame_equal(app.session_state["dataset"], original)


def test_classification_workflow_reaches_training() -> None:
    values, labels = make_classification(n_samples=40, n_features=3, n_informative=2, n_redundant=0, random_state=3)
    dataset = pd.DataFrame(values, columns=["a", "b", "c"])
    dataset["target"] = labels
    app = _seed_app(dataset, "target", "classification")
    _apply_and_train(app)
    assert app.session_state["preprocessing_applied"] is True
    assert "Logistic Regression" in app.session_state["trained_models"]
    assert app.session_state["best_pipeline"] is None


def test_regression_workflow_reaches_training() -> None:
    values, target = make_regression(n_samples=40, n_features=3, random_state=3)
    dataset = pd.DataFrame(values, columns=["a", "b", "c"])
    dataset["target"] = target
    app = _seed_app(dataset, "target", "regression")
    _apply_and_train(app)
    assert "Ridge" in app.session_state["trained_models"]
    assert app.session_state["training_results"]["Ridge"]["status"] == "trained"


def test_missing_categorical_workflow_reaches_training() -> None:
    dataset = pd.DataFrame(
        {
            "age": [20.0, 30.0, None, 40.0, 50.0, 60.0, 70.0, 80.0],
            "kind": ["A", "B", "A", None, "B", "A", "B", "A"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    app = _seed_app(dataset, "target", "classification")
    app.session_state["prep_selected_features"] = ["age", "kind"]
    app.session_state["prep_numeric_missing"] = "Median"
    app.session_state["prep_categorical_missing"] = "Most Frequent"
    _apply_and_train(app)
    assert "Logistic Regression" in app.session_state["trained_models"]


def test_preprocessing_change_invalidates_training() -> None:
    values, labels = make_classification(n_samples=40, n_features=3, n_informative=2, n_redundant=0, random_state=3)
    dataset = pd.DataFrame(values, columns=["a", "b", "c"])
    dataset["target"] = labels
    app = _seed_app(dataset, "target", "classification")
    _apply_and_train(app)
    preprocessing_nav = next(button for button in app.button if "Preprocessing" in button.label)
    preprocessing_nav.click().run()
    app.selectbox(key="prep_scaling").select("StandardScaler").run()
    assert not app.exception, app.exception
    assert app.session_state["preprocessing_applied"] is False
    assert not app.session_state["trained_models"]


def test_model_selection_change_invalidates_training_without_rerun_error() -> None:
    values, labels = make_classification(n_samples=40, n_features=3, n_informative=2, n_redundant=0, random_state=3)
    dataset = pd.DataFrame(values, columns=["a", "b", "c"])
    dataset["target"] = labels
    app = _seed_app(dataset, "target", "classification")
    _apply_and_train(app)
    app.multiselect(key="selected_models").set_value(["Random Forest"]).run()
    assert not app.exception, app.exception
    assert not app.session_state["trained_models"]


@pytest.mark.parametrize(
    ("problem_type", "target_values", "select_model"),
    [
        ("classification", [0, 1, 0, 1, 0, 1, 0, 1], "Logistic Regression"),
        ("regression", [10.0, 20.0, 25.0, 35.0, 42.0, 53.0, 61.0, 75.0], "Ridge"),
    ],
)
def test_models_ui_filters_registry_and_preserves_card_selection(
    problem_type: str,
    target_values: list[int] | list[float],
    select_model: str,
) -> None:
    dataset = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "group": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "target": target_values,
        }
    )
    features = ["value", "group"]
    config = PreprocessingConfig(
        selected_features=features,
        target_column="target",
        problem_type=problem_type,
    )
    pipeline = build_preprocessing_pipeline(config, ["value"], ["group"])
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=20)
    for key, value in {
        "current_page": "models",
        "dataset": dataset,
        "dataset_name": "models-ui.csv",
        "dataset_valid": True,
        "target_column": "target",
        "problem_type": problem_type,
        "detected_problem_type": problem_type,
        "selected_features": features,
        "preprocessing_pipeline": pipeline,
        "preprocessing_config": config.to_dict(),
        "preprocessing_applied": True,
        "selected_models": [],
        "train_test_size": 0.2,
        "train_random_state": 42,
    }.items():
        app.session_state[key] = value

    app.run()
    assert not app.exception, app.exception
    assert get_available_models(problem_type)
    assert get_model_families(problem_type)[0] == "All Families"
    assert any(select_model in item.value for item in app.markdown)
    assert any(item.label == "Search models" for item in app.text_input)

    app.selectbox(key="model_family_filter").select("Linear").run()
    assert not app.exception, app.exception
    app.text_input(key="model_search").set_value("no-matching-model").run()
    assert not app.exception, app.exception
    assert any("No models match" in item.value for item in app.info)
    app.text_input(key="model_search").set_value(select_model).run()
    assert not app.exception, app.exception
    assert any(select_model in item.value for item in app.markdown)

    toggle_key = f"model_toggle_{problem_type}_{select_model}"
    next(button for button in app.button if button.key == toggle_key).click().run()
    assert not app.exception, app.exception
    assert app.session_state["selected_models"] == [select_model]

    app.selectbox(key="model_family_filter").select("Tree-Based").run()
    assert not app.exception, app.exception
    assert app.session_state["selected_models"] == [select_model]


def test_regression_models_with_shared_parameter_controls_have_unique_widgets() -> None:
    dataset = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "target": [10.0, 15.0, 22.0, 28.0, 35.0, 41.0],
        }
    )
    config = PreprocessingConfig(
        selected_features=["value"],
        target_column="target",
        problem_type="regression",
    )
    pipeline = build_preprocessing_pipeline(config, ["value"], [])
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=20)
    for key, value in {
        "current_page": "models",
        "dataset": dataset,
        "dataset_name": "tree-models.csv",
        "dataset_valid": True,
        "target_column": "target",
        "problem_type": "regression",
        "detected_problem_type": "regression",
        "selected_features": ["value"],
        "preprocessing_pipeline": pipeline,
        "preprocessing_config": config.to_dict(),
        "preprocessing_applied": True,
        "selected_models": ["Decision Tree", "Decision Tree Regressor"],
    }.items():
        app.session_state[key] = value

    app.run()
    assert not app.exception, app.exception
    assert app.session_state["selected_models"] == ["Decision Tree", "Decision Tree Regressor"]


def _run_csv_training_workflow(
    filename: str,
    dataset: pd.DataFrame,
    expected_problem_type: str,
    model_name: str,
) -> AppTest:
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=30).run()
    app.file_uploader[0].upload(filename, dataset.to_csv(index=False).encode("utf-8"), "text/csv").run()
    assert not app.exception, app.exception
    target_selector = next(selectbox for selectbox in app.selectbox if selectbox.label == "Select target column")
    target_selector.select("target").run()
    assert not app.exception, app.exception
    assert app.session_state["problem_type"] == expected_problem_type

    next(button for button in app.button if "Continue to EDA" in button.label).click().run()
    assert not app.exception, app.exception
    next(button for button in app.button if "Let's do preprocessing" in button.label).click().run()
    assert not app.exception, app.exception
    next(button for button in app.button if "Apply preprocessing configuration" in button.label).click().run()
    assert not app.exception, app.exception
    assert app.session_state["preprocessing_applied"] is True
    next(button for button in app.button if "Configure models" in button.label).click().run()
    assert not app.exception, app.exception
    assert app.selectbox(key="train_test_size").value == 0.2
    assert app.number_input(key="train_random_state").value == 42

    app.multiselect(key="selected_models").set_value([model_name]).run()
    assert not app.exception, app.exception
    parameter_name, parameter_key, parameter_value = (
        ("C", "model_Logistic Regression_c", 2.5)
        if model_name == "Logistic Regression"
        else ("alpha", "model_ridge_alpha", 2.5)
    )
    app.number_input(key=parameter_key).set_value(parameter_value).run()
    app.selectbox(key="train_test_size").select(0.3).run()
    app.number_input(key="train_random_state").set_value(7).run()
    next(button for button in app.button if "Train selected models" in button.label).click().run()
    assert not app.exception, app.exception
    assert model_name in app.session_state["trained_models"]
    assert app.session_state["training_results"][model_name]["status"] == "trained"
    estimator = app.session_state["trained_models"][model_name]["pipeline"].named_steps["model"]
    assert estimator.get_params()[parameter_name] == parameter_value
    loaded_dataset = app.session_state["dataset"]
    _, expected_test, _, _ = split_dataset(
        loaded_dataset.drop(columns="target"),
        loaded_dataset["target"],
        expected_problem_type,
        test_size=0.3,
        random_state=7,
    )
    assert app.session_state["X_test"].equals(expected_test)
    return app


def test_real_breast_cancer_csv_completes_classification_workflow() -> None:
    frame = load_breast_cancer(as_frame=True).frame
    _run_csv_training_workflow("breast_cancer.csv", frame, "classification", "Logistic Regression")


def test_real_diabetes_csv_completes_regression_workflow() -> None:
    frame = load_diabetes(as_frame=True).frame
    _run_csv_training_workflow("diabetes.csv", frame, "regression", "Ridge")


def test_all_model_training_failures_are_shown_in_models_ui() -> None:
    dataset = pd.DataFrame(
        {
            "value": [-8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, -1.0],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    config = PreprocessingConfig(selected_features=["value"], target_column="target", problem_type="classification")
    pipeline = build_preprocessing_pipeline(config, ["value"], [])
    app = AppTest.from_file(Path(__file__).parents[1] / "app.py", default_timeout=20)
    for key, value in {
        "current_page": "models",
        "dataset": dataset,
        "dataset_name": "negative-counts.csv",
        "dataset_valid": True,
        "target_column": "target",
        "problem_type": "classification",
        "detected_problem_type": "classification",
        "selected_features": ["value"],
        "preprocessing_pipeline": pipeline,
        "preprocessing_config": config.to_dict(),
        "preprocessing_applied": True,
        "selected_models": ["Multinomial Naive Bayes"],
    }.items():
        app.session_state[key] = value

    app.run()
    next(button for button in app.button if "Train selected models" in button.label).click().run()

    assert not app.exception, app.exception
    assert not app.session_state["trained_models"]
    assert app.session_state["training_results"]["Multinomial Naive Bayes"]["status"] == "failed"
    assert any("None of the selected models trained successfully" in item.value for item in app.error)
