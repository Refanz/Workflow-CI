import os.path
import sys
import warnings

import mlflow
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Load dataset
    dataset_path = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                      "mushrooms_preprocessed.csv")
    mushroom_df = pd.read_csv(dataset_path)

    # Melakukan splitting dataset
    X = mushroom_df.drop(columns=['class'], axis=1)
    y = mushroom_df['class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Menyimpan sample input
    input_example = X_train[0:5]

    # Hyperparameter
    solver = sys.argv[1] if len(sys.argv) > 1 else "liblinear"
    penalty = sys.argv[2] if len(sys.argv) > 2 else "l1"
    C = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    # Melatih model dan menjalankan mlflow
    with mlflow.start_run():
        model = LogisticRegression(
            solver=solver,
            penalty=penalty,
            C=C,
            random_state=42,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
        )

        # Log metrics
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)