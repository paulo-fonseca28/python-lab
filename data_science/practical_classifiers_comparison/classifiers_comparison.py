import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


RANDOM_STATE = 42


# -------------------------
# Helpers
# -------------------------
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Strip em strings e converte '?' para NaN."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("?", np.nan)
            df[col] = df[col].replace("unknown", np.nan)
    return df


def choose_pos_label(classes: list[str], desired: str) -> str:
    """Garante que desired exista; senão escolhe uma classe positiva de fallback."""
    desired = str(desired)
    if desired in classes:
        return desired
    if len(classes) >= 2:
        return sorted(classes)[1]
    return classes[0]


def make_binary_target(y_str: pd.Series, desired_pos_label: str):
    """
    Converte alvo string em binário 0/1:
      y_bin = 1 para a classe positiva escolhida
      y_bin = 0 para a outra classe
    Retorna: y_bin, pos_label_str, neg_label_str, classes_str
    """
    classes = sorted(pd.Series(y_str).dropna().astype(str).unique().tolist())
    if len(classes) != 2:
        raise ValueError(f"Este script assume classificação binária. Classes encontradas: {classes}")

    pos_label_str = choose_pos_label(classes, desired_pos_label)
    neg_label_str = classes[0] if classes[1] == pos_label_str else classes[1]

    y_bin = (y_str.astype(str) == pos_label_str).astype(int)
    return y_bin, pos_label_str, neg_label_str, classes


def build_preprocess(X: pd.DataFrame) -> ColumnTransformer:
    num_selector = make_column_selector(dtype_include=np.number)
    cat_selector = make_column_selector(dtype_exclude=np.number)

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_selector),
            ("cat", categorical_pipe, cat_selector),
        ],
        remainder="drop",
    )


def eval_binary(model: Pipeline, X_test: pd.DataFrame, y_test_bin: pd.Series):
    """Avalia com y binário (0/1), positivo = 1."""
    y_true = y_test_bin.to_numpy()
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_true, proba)
    else:
        auc = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return acc, f1, auc, cm


# -------------------------
# Nested CV core
# -------------------------
def nested_stratified_cv(
    df: pd.DataFrame,
    target_col: str,
    dataset_name: str,
    desired_pos_label: str,
    outer_splits: int = 5,
    inner_splits: int = 3,
    n_jobs: int = -1,
):
    df = clean_dataframe(df)

    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = (
            df[target_col]
            .astype(str)
            .str.strip()
            .str.replace(".", "", regex=False)
        )

    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    X = df.drop(columns=[target_col])
    y_str = df[target_col].astype(str)

    y_bin, pos_label_str, neg_label_str, classes_str = make_binary_target(y_str, desired_pos_label)

    print("\n" + "=" * 100)
    print(f"[{dataset_name}] Nested Stratified CV | outer={outer_splits} inner={inner_splits}")
    print(f"Shape: {df.shape} | Target: {target_col}")
    print(f"Classes originais: {classes_str}")
    print(f"Classe positiva (vira 1): {pos_label_str}")
    print(f"Classe negativa (vira 0): {neg_label_str}")
    print("Distribuição (binária):\n", pd.Series(y_bin).value_counts().sort_index().rename(index={0:"neg(0)",1:"pos(1)"}))
    print("=" * 100 + "\n")

    preprocess = build_preprocess(X)

    specs = [
        (
            "KNN",
            Pipeline(steps=[
                ("preprocess", preprocess),
                ("model", KNeighborsClassifier())
            ]),
            {
                "model__n_neighbors": [5, 15, 31],
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],  # Manhattan vs Euclidiana
            },
        ),
        (
            "DecisionTree",
            Pipeline(steps=[
                ("preprocess", preprocess),
                ("model", DecisionTreeClassifier(
                    random_state=RANDOM_STATE,
                    class_weight="balanced"
                ))
            ]),
            {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [None, 20],
                "model__min_samples_leaf": [1, 5],
            },
        ),
        (
            "RandomForest",
            Pipeline(steps=[
                ("preprocess", preprocess),
                ("model", RandomForestClassifier(
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    class_weight="balanced_subsample"
                ))
            ]),
            {
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 20],
                "model__min_samples_leaf": [1, 5],
            },
        ),
    ]

    outer_cv = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=RANDOM_STATE)

    fold_rows = []

    scoring = {
        "f1": "f1",
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
    }

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y_bin), start=1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y_bin.iloc[tr_idx], y_bin.iloc[te_idx]

        for model_name, pipe, grid in specs:
            gs = GridSearchCV(
                estimator=pipe,
                param_grid=grid,
                scoring=scoring,
                refit="f1",
                cv=inner_cv,
                n_jobs=n_jobs,
                verbose=0,
                error_score="raise",
                return_train_score=False
            )

            gs.fit(X_tr, y_tr)
            best = gs.best_estimator_

            acc, f1v, auc, cm = eval_binary(best, X_te, y_te)

            fold_rows.append({
                "dataset": dataset_name,
                "fold": fold,
                "model": model_name,
                "pos_label_str": pos_label_str,
                "neg_label_str": neg_label_str,
                "best_params": str(gs.best_params_),
                "inner_best_f1": float(gs.best_score_),
                "test_accuracy": float(acc),
                "test_f1": float(f1v),
                "test_roc_auc": float(auc),
                "cm_00": int(cm[0, 0]),
                "cm_01": int(cm[0, 1]),
                "cm_10": int(cm[1, 0]),
                "cm_11": int(cm[1, 1]),
            })

            print(f"[{dataset_name}] fold={fold} | {model_name}")
            print("  best_params:", gs.best_params_)
            print(f"  inner_best_f1={gs.best_score_:.4f} | test: acc={acc:.4f} f1={f1v:.4f} auc={auc:.4f}")
            print(f"  CM (0=neg,1=pos): [[{cm[0,0]},{cm[0,1]}],[{cm[1,0]},{cm[1,1]}]]")
            print()

    folds_df = pd.DataFrame(fold_rows)

    summary = (
        folds_df
        .groupby(["dataset", "model"], as_index=False)
        .agg(
            mean_accuracy=("test_accuracy", "mean"),
            std_accuracy=("test_accuracy", "std"),
            mean_f1=("test_f1", "mean"),
            std_f1=("test_f1", "std"),
            mean_auc=("test_roc_auc", "mean"),
            std_auc=("test_roc_auc", "std"),
        )
        .sort_values(["dataset", "mean_f1"], ascending=[True, False])
    )

    return folds_df, summary


# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    adult_path = base_dir / "adult.csv"
    bank_path = base_dir / "bank.csv"

    adult_df = pd.read_csv(adult_path)
    bank_df = pd.read_csv(bank_path)

    adult_target = "income"
    bank_target = "deposit"

    folds_adult, summary_adult = nested_stratified_cv(
        adult_df, adult_target, "Adult Census Income", desired_pos_label=">50K",
        outer_splits=5, inner_splits=3, n_jobs=-1
    )

    folds_bank, summary_bank = nested_stratified_cv(
        bank_df, bank_target, "Bank Marketing", desired_pos_label="yes",
        outer_splits=5, inner_splits=3, n_jobs=-1
    )

    folds_all = pd.concat([folds_adult, folds_bank], ignore_index=True)
    summary_all = pd.concat([summary_adult, summary_bank], ignore_index=True)

    print("\n" + "=" * 100)
    print("RESUMO FINAL (média ± std nos folds externos):")
    print(summary_all.to_string(index=False))
    print("=" * 100 + "\n")

    folds_all.to_csv(base_dir / "nestedcv_results_folds.csv", index=False)
    summary_all.to_csv(base_dir / "nestedcv_results_summary.csv", index=False)
    print("Arquivos gerados:")
    print(" - nestedcv_results_folds.csv")
    print(" - nestedcv_results_summary.csv")