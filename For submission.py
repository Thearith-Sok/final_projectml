import pandas as pd
import warnings
import numpy as np

# Filter out warnings for cleaner output
warnings.filterwarnings("ignore") 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

class SleepQualityPredictor:
    def __init__(self):
        self.imputer = SimpleImputer(strategy="median")
        self.models = {}
        self.best_model = None
        self.best_name = None
        self.preprocessor = None
        self.numeric_features = None
        self.categorical_features = None

    def load_data(self, path):
        """Loads and cleans the initial data."""
        df = pd.read_csv(path)
        # Drop irrelevant columns
        if "Blood Pressure" in df.columns:
            df = df.drop(columns=["Blood Pressure", "Person ID", "Sleep Disorder"])
        return df

    def preprocess(self, df):
        """
        Sets up the ColumnTransformer for preprocessing (Imputing, Scaling, and One-Hot Encoding) on the training data.
        """
        X = df.drop(columns=["Quality of Sleep"])
        y = df["Quality of Sleep"]

        # Identify feature types
        self.numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        self.categorical_features = X.select_dtypes(include="object").columns.tolist()

        # Create preprocessing pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')) 
        ])

        # Combine transformers
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='passthrough'
        )
        
        # Fit and transform the data
        X_processed = self.preprocessor.fit_transform(X)

        return X_processed, y, X.columns.tolist()

    def train(self, X, y):
        """
        Trains models using GridSearchCV for hyperparameter tuning and cross-validation 
        to fight overfitting and select the best model.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Define Models and Parameter Grids for tuning
        candidates = {
            "Random Forest": {
                "model": RandomForestRegressor(random_state=42),
                "params": {
                    'n_estimators': [50, 100],
                    'max_depth': [5, 10],
                }
            },
            "Gradient Boosting": {
                "model": GradientBoostingRegressor(random_state=42),
                "params": {
                    'n_estimators': [50, 100],
                    'learning_rate': [0.05, 0.1],
                }
            },
            "Linear Regression": {
                "model": LinearRegression(),
                "params": {}
            }
        }

        best_mae = float("inf")

        for name, candidate in candidates.items():
            model = candidate["model"]
            params = candidate["params"]
            
            # Use GridSearchCV for Hyperparameter Tuning with 5-fold Cross-Validation (CV)
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params,
                scoring='neg_mean_absolute_error', # Maximize negative MAE (minimize MAE)
                cv=5, 
                verbose=0,
                n_jobs=-1
            )

            grid_search.fit(X_train, y_train)
            
            preds = grid_search.best_estimator_.predict(X_test)
            mae = mean_absolute_error(y_test, preds)
            
            self.models[name] = grid_search.best_estimator_

            print(f"{name}: Best Params = {grid_search.best_params_}")
            print(f"{name}: MAE = {mae:.3f}")

            if mae < best_mae:
                best_mae = mae
                self.best_model = grid_search.best_estimator_
                self.best_name = name

        print(f"\nBest Model = {self.best_name} (MAE={best_mae:.3f})")
        return X_test, y_test

    def get_feature_importance(self):
        """Calculates and returns a DataFrame of feature importances."""
        if self.best_model and hasattr(self.best_model, 'feature_importances_'):
            
            # Get the feature names after one-hot encoding
            onehot_encoder = self.preprocessor.named_transformers_['cat']['onehot']
            onehot_features = list(onehot_encoder.get_feature_names_out(self.categorical_features))
            
            # Combine all feature names
            all_feature_names = self.numeric_features + onehot_features

            importance = self.best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': importance.round(6) # Rounding for cleaner presentation
            }).sort_values(by='Importance', ascending=False)
            
            return feature_importance_df
        return None

# USER INPUT SIMPLIFIED
def get_category(label):
    return input(f"{label}: ").strip()

def get_user_input(features, predictor):
    """Collects raw user input for all features and returns a DataFrame."""
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
        if col in predictor.categorical_features:
            user[col] = get_category(prompts.get(col, col))
        else:
            try:
                user[col] = float(input(f"{prompts.get(col, col)}: "))
            except ValueError:
                # Fallback for numerical input errors
                user[col] = 0

    return pd.DataFrame([user])


# INTERPRETATION
def interpret(score):
    """Maps continuous score to categorical label."""
    if score >= 9:
        return "Excellent! Your sleep habits are very healthy. Keep maintaining consistent sleep routines."

    if score >= 7:
        return ("Good. Your sleep quality is above average, but you could still improve by "
                "maintaining a consistent sleep schedule and reducing screen time before bed.")

    if score >= 5:
        return ("Fair. Consider improving your bedtime routine, reducing stress, and avoiding caffeine in the evening.")

    return ("Poor (needs improvement). Try to increase sleep duration, lower stress levels, "
            "exercise regularly, and avoid heavy meals before bedtime.")

def get_personalized_suggestions(user):
    """
    Takes the raw user input DataFrame (single row) and generates personalized advice.
    Returns a list of suggestions.
    """
    u = user.iloc[0]  # Get the row easily
    suggestions = []

    # Sleep Duration
    if u["Sleep Duration"] < 6:
        suggestions.append("Try to sleep at least 7–8 hours per night for better recovery.")
    elif u["Sleep Duration"] > 9:
        suggestions.append("Long sleep duration may indicate fatigue — ensure consistent sleep scheduling.")

    # Stress Level
    if u["Stress Level"] >= 7:
        suggestions.append("Your stress level is high. Try meditation, deep breathing exercises, or reducing evening workload.")
    elif u["Stress Level"] <= 3:
        suggestions.append("Great job managing stress — keep maintaining relaxing routines.")

    # Physical Activity
    if u["Physical Activity Level"] <= 3:
        suggestions.append("Increase your physical activity to at least moderate levels (20–30 min daily).")
    elif u["Physical Activity Level"] >= 8:
        suggestions.append("You're extremely active — ensure proper rest and hydration.")

    # Heart Rate
    if u["Heart Rate"] > 90:
        suggestions.append("Your heart rate is high — try relaxation before bed such as light stretching or avoiding caffeine.")
    elif u["Heart Rate"] < 50:
        suggestions.append("Low heart rate often indicates fitness, but if you feel tired, monitor your sleep recovery.")

    # Daily Steps
    if u["Daily Steps"] < 4000:
        suggestions.append("Try increasing your daily steps (aim for at least 6,000–7,000).")
    elif u["Daily Steps"] > 12000:
        suggestions.append("Great activity level — just ensure you're getting adequate rest.")

    # BMI Category
    if u["BMI Category"] == "Overweight" or u["BMI Category"] == "Obese":
        suggestions.append("Improving diet and doing light exercise may help improve sleep quality.")
    elif u["BMI Category"] == "Underweight":
        suggestions.append("Ensure proper nutrition — being underweight can also affect sleep quality.")

    # Occupation (optional creativity boost)
    if u["Occupation"] in ["Nurse", "Doctor", "Security", "Driver"] or \
       "shift" in u["Occupation"].lower():
        suggestions.append("Shift-work can reduce sleep quality — try maintaining consistent sleep windows on off days.")

    # Gender-specific example (optional)
    if u["Gender"].lower() == "female":
        suggestions.append("Women often experience sleep disturbances during hormonal cycles — consistent routines can help.")

    # If no suggestions matched
    if not suggestions:
        suggestions.append("Your lifestyle looks balanced — keep maintaining healthy sleep habits!")

    return suggestions

# MAIN PROGRAM
def main():
    predictor = SleepQualityPredictor()
    
    # Train Models and Select Best
    predictor.train(X_processed, y)

    # Feature Importance Analysis
    if predictor.best_model and predictor.best_name in ["Random Forest", "Gradient Boosting"]:
        importance_df = predictor.get_feature_importance()
        if importance_df is not None:
            print("\n--- Feature Importance (Top 5) ---")
            print(importance_df.head(5).to_markdown(index=False))

    # User Prediction Loop
    while True:
        print("\n--- Quality of Sleep Predictor ---")
        
        # Get raw user input using original feature names
        user_df = get_user_input(original_feature_names, predictor) 

        # Transform raw user input using the fitted preprocessor
        user_processed = predictor.preprocessor.transform(user_df)

        # Predict
        pred = predictor.best_model.predict(user_processed)[0]

        # Clamp prediction to the 1-10 range
        pred = max(1, min(10, pred))

        print(f"\nPredicted Sleep Quality: {pred:.1f}/10")
        print("Interpretation:", interpret(pred))

        # Personalized suggestions
        suggestions = get_personalized_suggestions(user_df)

        print("\nPersonalized Suggestions:")
        for s in suggestions:
            print("• " + s)

        if input("\nTry again? (y/n): ").lower() != "y":
            break


if __name__ == "__main__":
    main()
