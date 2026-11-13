
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

cols = ["unit", "cycle", "set1", "set2", "set3"] + [
    "s" + str(i) for i in range(1, 22)
]

df = pd.read_csv(
    "train_FD001.txt",
    sep=r"\s+",
    header=None,
    names=cols
)

print("Dataset shape:", df.shape)

# Select engine 1
df = df[df["unit"] == 1].copy()
df = df.dropna(subset=["cycle", "s4"])

# Input and output
X = df[["cycle"]]
y = df["s4"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))

# Predict at cycle 250
prediction = model.predict(pd.DataFrame({"cycle": [250]}))
print("Predicted T50 at cycle 250:", prediction[0])

# Plot graph
plt.scatter(X_test["cycle"], y_test, label="Actual T50")
plt.plot(
    X_test["cycle"],
    y_pred,
    color="red",
    label="Predicted T50"
)

plt.xlabel("Operating Cycles")
plt.ylabel("Turbine Temperature (T50)")
plt.title("Engine Wear: Turbine Temperature Rises with Age")
plt.legend()
plt.show()
