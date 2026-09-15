import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the data (skipping bad lines to avoid the previous ParserError)
print("Loading data...")
df = pd.read_csv('data/hums_data2.csv', on_bad_lines='skip')

# 2. Define your exact column names based on your CSV:
# If your CSV headers are spelled differently, please update them here!

features = ['engine_temp_c', 'vibration_mm_s'] 
target = 'status'

# Column validation check
missing_cols = [col for col in features + [target] if col not in df.columns]
if missing_cols:
    print(f"Error: Could not find these columns in the CSV: {missing_cols}")
    print(f"Columns available in your CSV: {list(df.columns)}")
    print("Please fix the 'features' and 'target' variable names in the code.")
    exit()

# 3. Prepare the data (remove empty rows)
df = df.dropna(subset=features + [target])
X = df[features]
y = df[target]

# Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Test the model and display the accuracy
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the trained model to use in your Streamlit dashboard
joblib.dump(model, 'predictive_model.pkl')
print("Success! 'predictive_model.pkl' has been generated.")
