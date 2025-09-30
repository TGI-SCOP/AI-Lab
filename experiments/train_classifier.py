from pathlib import Path
import json
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def main():
    X, y = load_iris(return_X_y=True, as_frame=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, clf.predict(X_te))

    Path("models").mkdir(parents=True, exist_ok=True)
    Path("experiments").mkdir(parents=True, exist_ok=True)

    # Sauvegarde "modèle" (pickle) + métriques (json)
    import pickle
    with open("models/iris_logreg.pkl", "wb") as f:
        pickle.dump(clf, f)

    result = {
        "experiment": "iris_logreg",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "accuracy": acc
    }
    with open("experiments/iris_logreg_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"Test accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
