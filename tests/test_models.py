import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.models.model_factory import create_model
from src.models.registry import get_available_models, get_model_families, get_model_metadata, get_models_by_family


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Logistic Regression", LogisticRegression),
        ("Naive Bayes", GaussianNB),
        ("K-Nearest Neighbors (KNN)", KNeighborsClassifier),
        ("Support Vector Machine (SVM)", SVC),
        ("Decision Tree", DecisionTreeClassifier),
        ("Random Forest", RandomForestClassifier),
        ("Extra Trees", ExtraTreesClassifier),
        ("AdaBoost", AdaBoostClassifier),
        ("Gradient Boosting", GradientBoostingClassifier),
        ("HistGradientBoosting", HistGradientBoostingClassifier),
    ],
)
def test_classification_model_factory(name: str, expected: type) -> None:
    assert isinstance(create_model(name, "classification"), expected)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Linear Regression", LinearRegression),
        ("Ridge Regression", Ridge),
        ("Lasso Regression", Lasso),
        ("Elastic Net", ElasticNet),
        ("K-Nearest Neighbors Regressor", KNeighborsRegressor),
        ("Support Vector Regressor (SVR)", SVR),
        ("Decision Tree Regressor", DecisionTreeRegressor),
        ("Random Forest Regressor", RandomForestRegressor),
        ("Extra Trees Regressor", ExtraTreesRegressor),
        ("AdaBoost Regressor", AdaBoostRegressor),
        ("Gradient Boosting Regressor", GradientBoostingRegressor),
        ("HistGradientBoosting Regressor", HistGradientBoostingRegressor),
    ],
)
def test_regression_model_factory(name: str, expected: type) -> None:
    assert isinstance(create_model(name, "regression"), expected)


def test_registry_filters_by_problem_type() -> None:
    classification = get_available_models("classification")
    regression = get_available_models("regression")
    assert "Support Vector Machine (SVM)" in classification
    assert "Support Vector Regressor (SVR)" in regression
    assert "Support Vector Machine (SVM)" not in regression
    assert "Support Vector Regressor (SVR)" not in classification
    assert "Decision Tree" in classification and "Decision Tree Regressor" in regression
    assert len(classification) == 10
    assert len(regression) == 12


def test_registry_exposes_dynamic_family_filters() -> None:
    families = get_model_families("classification")
    assert "All Families" in families
    assert "Boosting" in families
    assert all(get_model_metadata(name).problem_type in {"classification", "both"} for name in get_models_by_family("classification", "Boosting"))


@pytest.mark.parametrize("problem_type", ["classification", "regression"])
def test_all_registry_models_can_be_instantiated(problem_type: str) -> None:
    for model_name in get_available_models(problem_type):
        assert create_model(model_name, problem_type) is not None


def test_model_parameters_are_passed() -> None:
    model = create_model("Random Forest", "classification", {"n_estimators": 7, "max_depth": 3})
    assert model.n_estimators == 7
    assert model.max_depth == 3


def test_invalid_model_name_and_problem_type_raise() -> None:
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_metadata("Unknown")
    with pytest.raises(ValueError, match="not available"):
        create_model("Support Vector Regressor (SVR)", "classification")
