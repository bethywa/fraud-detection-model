from sklearn.ensemble import RandomForestClassifier
import joblib

def train_random_forest(X_train, y_train, random_state=42):
    model = RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, path):
    joblib.dump(model, path)
