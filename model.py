print("Script started!")
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
import pickle

# Load data
df = pd.read_csv("ds_salaries.csv")

# Encode text columns into numbers
le = LabelEncoder()
df['experience_level'] = le.fit_transform(df['experience_level'])
df['job_title'] = le.fit_transform(df['job_title'])
df['company_size'] = le.fit_transform(df['company_size'])

# Pick columns to learn from
X = df[['experience_level', 'job_title', 'company_size', 'remote_ratio']]
y = df['salary_in_usd']

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Check accuracy
preds = model.predict(X_test)
print("Error:", mean_absolute_error(y_test, preds))

# Save the model
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved!")