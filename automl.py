import itertools
import time
import os
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,make_scorer

# Load Data
df = pd.read_csv('Dataset of Diabetes .csv')
df.info()

x = df.drop(columns=['CLASS','ID', 'No_Pation'])
y = df["CLASS"].astype(str).str.strip().str.upper()

print("\n===== TARGET CLASS DISTRIBUTION =====")
print(y.value_counts())
print("\n===== UNIQUE TARGET VALUES =====")
print(y.unique())

print("\nFeatures:")
print(x.columns.tolist())

# Checking missing values in features
print("\n===== MISSING VALUES =====")

print(x.isna().sum())
print(y.isnull().sum())

for col in x.select_dtypes(include=['object', 'category']).columns:
    x[col] = x[col].astype(str).str.strip().str.upper()

categorical_columns = x.select_dtypes(['object', 'category']).columns.tolist()
numerical_columns = x.select_dtypes(['number']).columns.tolist()
boolean_columns = x.select_dtypes(['bool']).columns.tolist()

for column in boolean_columns:
    x[column] = x[column].astype('int')

print("\n===== COLUMN TYPES =====")
print("Categorical:", categorical_columns)
print("Numerical:", numerical_columns)
print("Boolean:", boolean_columns)

# ============================================================
# PREPARE DATA AND PIPELINE
# ============================================================

def prepare_data_and_pipeline(x, y, categorical_columns, numerical_columns, boolean_columns):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2,
        random_state=42,
        stratify=y)

    preprocessing = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ],
        remainder='passthrough')

    return (x_train, x_test, y_train, y_test, preprocessing)

# ============================================================
# MODEL CREATION
# ============================================================
def create_model(model_name, params, preprocessor):
    if model_name == 'LogisticRegression':
        clf = LogisticRegression(max_iter=1000,
                                 random_state=42,
                                 class_weight="balanced",
                                 **params)
    elif model_name == 'DecisionTreeClassifier':
        clf = DecisionTreeClassifier(random_state=42,
                                     class_weight="balanced",
                                     **params)
    elif model_name == 'RandomForestClassifier':
        clf = RandomForestClassifier(random_state=42,
                                     class_weight="balanced",
                                     **params)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    #Complete pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', clf)
    ])
    return pipeline

# ============================================================
# MODEL TUNING
# ============================================================
def tune_model(pipeline, param_grid, x_train, y_train):
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1,
    )
    grid_search.fit(x_train, y_train)

    print(f"Best CV Score (Accuracy): {grid_search.best_score_:.4f}")
    print("Best Parameters:", grid_search.best_params_)

    return grid_search.best_estimator_, grid_search.best_score_


# Unpack variables for the initial GridSearchCV Loop
x_train, x_test, y_train, y_test, preprocessor = prepare_data_and_pipeline(
    x, y, categorical_columns, numerical_columns, boolean_columns)

# Define the hyperparameter Search Grid
meta_param_grids = {
    'LogisticRegression': {
        'model__C': [0.1, 1.0, 10.0]
    },
    'DecisionTreeClassifier': {
        'model__max_depth': [5, 10, 20, None],
        'model__min_samples_split': [2, 5, 10]
    },
    'RandomForestClassifier': {
        'model__n_estimators':[50,100],
        'model__max_depth': [10, 20, None]
}
}

best_overall_score = -1
best_overall_pipeline = None
best_model_name = "None"

print("\n----- Starting Automl Evaluation Loop -----")

for model_name, param_grid in meta_param_grids.items():
    print(f"\nTuning Model: {model_name}...")
    base_pipeline = create_model(model_name, {}, preprocessor)
    tuned_pipeline, current_best_score = tune_model(base_pipeline, param_grid, x_train, y_train)

    print(f"-> {model_name} Final Best CV Accuracy: {current_best_score:.4f}")

    if current_best_score > best_overall_score:
        best_overall_score = current_best_score
        best_overall_pipeline = tuned_pipeline
        best_model_name = model_name

print("\n==============================================")
print(f"🏆 WINNING ARCHITECTURE: {best_model_name}")
print(f"🏆 ULTIMATE CROSS-VALIDATION Accuracy: {best_overall_score:.4f}")
print("==============================================")

# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(model, x_train, y_train, x_test, y_test):
    start_time = time.time()
    model.fit(x_train, y_train)
    training_time = time.time() - start_time

    # Prediction & Probabilities
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)

    # Multiclass ROC-AUC
    roc_auc = roc_auc_score(y_test,probabilities,multi_class='ovr',average='weighted')

    # Metrics Layout (Fixed: evaluate binary scores using predictions, NOT probability floats!)
    metrics_result = {
        "accuracy": accuracy_score(y_test, predictions),
        "roc_auc": roc_auc,
        "precision": precision_score(y_test, predictions, zero_division=0, average='weighted'),
        "recall": recall_score(y_test, predictions, zero_division=0, average='weighted'),
        "f1": f1_score(y_test, predictions, zero_division=0, average='weighted'),
        "training_time": training_time,
    }

    return metrics_result

def generate_combinations(parameter_space):
    """Converts a parameter grid dictionary into a list of explicit combination dicts."""
    if not parameter_space:
        return [{}]
    keys, values = zip(*parameter_space.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

# AutoML Engine
def run_automl(selected_models, search_spaces, progress_callback=None):
    # Fixed: Unpacking matches the exact signature sequence: (x_train, x_test, y_train, y_test)
    (x_train_local, x_test_local,
     y_train_local, y_test_local,
     preprocessor_local) = prepare_data_and_pipeline(x, y, categorical_columns, numerical_columns,boolean_columns)

    all_results = []
    total_experiments = 0

    # Calculate Total Experiments
    for model_name in selected_models:
        parameter_space = search_spaces.get(model_name, {})
        combinations = generate_combinations(parameter_space)
        total_experiments += len(combinations)

    completed_experiments = 0

    # Model Loop
    for model_name in selected_models:
        parameter_space = search_spaces.get(model_name, {})
        combinations = generate_combinations(parameter_space)

        print(f"\n🚀 Running Custom Engine Framework: {model_name}")

        # Fixed: Indented this entire block by 4 spaces to place it within the core model loop turn
        for params in combinations:
            print(f"  Testing Parameters: {params}")

            try:
                model = create_model(model_name, params, preprocessor_local)

                # Fixed: Uses local datasets in correct structural layout: (model, train_x, train_y, test_x, test_y)
                metrics = evaluate_model(model, x_train_local, y_train_local, x_test_local, y_test_local)

                result = {
                    "model": model_name,
                    "parameters": params,
                    **metrics
                }
                all_results.append(result)

            except Exception as e:
                print(f"  ❌ Experiment Failed: {e}")
                error_result = {
                    "model": model_name,
                    "parameters": params,
                    "error": str(e)
                }
                all_results.append(error_result)

            finally:
                completed_experiments += 1
                if progress_callback:
                    progress_callback(
                        completed_experiments,
                        total_experiments,
                        model_name,
                        params
                    )

    # Filter out errors and compile Leaderboard rankings
    valid_results = [result for result in all_results if "f1" in result]
    valid_results.sort(key=lambda x: x["f1"], reverse=True)

    for rank, result in enumerate(valid_results, start=1):
        result["rank"] = rank

    return valid_results


# =====================================================================
# SYSTEM EXECUTION WORKFLOW
# =====================================================================

def print_progress(completed, total, model_name, params):
    percentage = (completed / total) * 100
    print(f"   📊 Progress: [{completed}/{total}] {percentage:.1f}% Complete | Last evaluated: {model_name}")


custom_search_spaces = {
    'LogisticRegression': {
        'C': [0.1, 1.0]
    },
    'DecisionTreeClassifier': {
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    },
    'RandomForestClassifier': {
        'n_estimators':[50,100],
    'max_depth': [10, None]
}
}

models_to_test = ['LogisticRegression', 'DecisionTreeClassifier', 'RandomForestClassifier']

print("\n\n==============================================")
print("🚀 INITIALISING CUSTOM ENGINE AUTOMATION")
print("==============================================")

leaderboard = run_automl(
    selected_models=models_to_test,
    search_spaces=custom_search_spaces,
    progress_callback=print_progress
)
print()

print("\n==============================================")
print("🏆 FINAL CUSTOM ENGINE LEADERBOARD RANKINGS 🏆")
print("==============================================")
if leaderboard:
    leaderboard_df = pd.DataFrame(leaderboard)
    columns_order = ['rank', 'model', 'f1', 'roc_auc', 'precision', 'recall', 'accuracy', 'training_time', 'parameters']
    print(leaderboard_df[columns_order].to_string(index=False))
else:
    print("No experiments completed successfully.")

# =====================================================================
#  MODEL PACKAGING & EXPORT 
# =====================================================================
if leaderboard:
    best_run = leaderboard[0]
    winning_model_name = best_run["model"]
    winning_f1_score = best_run["f1"]
    winning_params = best_run["parameters"]

    print(f"\n📦 Packaging the optimal architecture configuration...")
    winning_pipeline = create_model(winning_model_name, winning_params, preprocessor)
    winning_pipeline.fit(x_train, y_train)

    model_filename = f"best_{winning_model_name.lower()}_f1_{winning_f1_score:.4f}.joblib"

    try:
        joblib.dump(winning_pipeline, model_filename)
        print(f"💾 EXPORT SUCCESSFUL: Pipeline architecture saved to '{model_filename}'")
    except Exception as e:
        print(f"❌ Export Failed: Could not save model file. Error: {e}")
else:
    print("\n⚠️ Export Skipped: No metrics returned to identify a winning framework.")