import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression


# ---------------------------------------------------------
# SLEEP QUALITY PREDICTOR 
# ---------------------------------------------------------
class SleepQualityPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="median")
        self.label_encoders = {}
        self.models = {}
        self.best_model = None

    # -----------------------------
    def load_data(self, path):
        df = pd.read_csv(path)
        # print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        # REMOVE BLOOD PRESSURE, Person ID, and Sleep Disorder
        if "Blood Pressure" in df.columns:
            df = df.drop(columns=["Blood Pressure", "Person ID", "Sleep Disorder"])

        return df

    # -----------------------------
    def preprocess(self, df):
        df = df.copy()

        # Encode all object columns except target
        for col in df.select_dtypes(include="object"):
            if col != "Quality of Sleep":
                le = LabelEncoder()
                self.label_encoders[col] = le
                df[col] = le.fit_transform(df[col].astype(str))

        X = df.drop(columns=["Quality of Sleep"])
        y = df["Quality of Sleep"]

        # Impute missing values
        X = pd.DataFrame(self.imputer.fit_transform(X), columns=X.columns)

        return X, y

    # -----------------------------
    def train(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Models
        candidates = {
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Linear Regression": LinearRegression()
        }

        best_mae = float("inf")

        for name, model in candidates.items():
            # Scale only for linear regression
            if name == "Linear Regression":
                X_train_s = self.scaler.fit_transform(X_train)
                X_test_s = self.scaler.transform(X_test)
                preds = model.fit(X_train_s, y_train).predict(X_test_s)
            else:
                preds = model.fit(X_train, y_train).predict(X_test)

            mae = mean_absolute_error(y_test, preds)
            self.models[name] = model

            # print(f"{name}: MAE = {mae:.3f}")

            if mae < best_mae:
                best_mae = mae
                self.best_model = model
                best_name = name

        print(f"\nBest Model = {best_name} (MAE={best_mae:.3f})")
        return X_test, y_test


# ---------------------------------------------------------
# USER INPUT SIMPLIFIED
# ---------------------------------------------------------
def get_category(label):
    return input(f"{label}: ").strip()

def get_user_input(features, encoders):
    print("\nEnter your information:")
    user = {}

    prompts = {
        "Age": "Age (years)",
        "Sleep Duration": "Sleep Duration (hours)",
        "Stress Level": "Stress Level (1-10)",
        "Physical Activity Level": "Physical Activity Level (1-10)",
        "Heart Rate": "Heart Rate (bpm)",
        "Daily Steps": "Daily Steps",
        "Gender": "Gender (Male/Female/Other)",
        "Occupation": "Occupation",
        "BMI Category": "BMI Category (Underweight/Normal/Overweight/Obese)"
    }

    for col in features:
        if col in encoders:  
            value = get_category(prompts.get(col, col))
            # unseen category handling
            if value in encoders[col].classes_:
                user[col] = encoders[col].transform([value])[0]
            else:
                user[col] = 0
        else:
            user[col] = float(input(f"{prompts.get(col, col)}: "))

    return pd.DataFrame([user])


# ---------------------------------------------------------
# INTERPRETATION
# ---------------------------------------------------------
def interpret(score):
    if score >= 9: return "Excellent!"
    if score >= 7: return "Good"
    if score >= 5: return "Fair"
    return "Poor (needs improvement)"


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------
def main():
    predictor = SleepQualityPredictor()

    df = predictor.load_data("Sleep_health_and_lifestyle_dataset.csv")
    X, y = predictor.preprocess(df)
    X_test, y_test = predictor.train(X, y)

    # User prediction loop
    while True:
        print("\n--- Quality of Sleep Predictor ---")
        user_df = get_user_input(X.columns.tolist(), predictor.label_encoders)

        # Scale 
        if isinstance(predictor.best_model, LinearRegression):
            user_df_scaled = predictor.scaler.transform(user_df)
            pred = predictor.best_model.predict(user_df_scaled)[0]
        else:
            pred = predictor.best_model.predict(user_df)[0]

        pred = max(1, min(10, pred))

        print(f"\nPredicted Sleep Quality: {pred:.1f}/10")
        print("Interpretation:", interpret(pred))

        if input("\nTry again? (y/n): ").lower() != "y":
            break


if __name__ == "__main__":
    main()
