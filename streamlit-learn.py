import streamlit as st
import pandas as pd
import numpy as np
import pickle

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="Iris ML Deployment",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Iris Machine Learning Deployment App")

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
class_names = iris.target_names

# --------------------------------------------------
# Sidebar Settings
# --------------------------------------------------
st.sidebar.header("⚙ Model Settings")

model_name = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Random Forest", "SVM", "KNN", "Naive Bayes"]
)

test_size = st.sidebar.slider("Test Size (%)", 10, 50, 30)
random_state = st.sidebar.number_input("Random State", value=42)

# --------------------------------------------------
# Model Factory
# --------------------------------------------------
@st.cache_resource
def get_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=200)
    elif name == "Random Forest":
        return RandomForestClassifier()
    elif name == "SVM":
        return SVC(probability=True)
    elif name == "KNN":
        return KNeighborsClassifier()
    elif name == "Naive Bayes":
        return GaussianNB()

model = get_model(model_name)

# --------------------------------------------------
# Train Model
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=test_size / 100,
    random_state=random_state
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# --------------------------------------------------
# Tabs Layout
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Model Evaluation", "🔍 Predict", "📁 Dataset"])

# --------------------------------------------------
# TAB 1 — Evaluation
# --------------------------------------------------
with tab1:

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Performance")
        st.metric("Accuracy", f"{accuracy:.2%}")

        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

    with col2:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    # Feature importance (only if RF)
    if model_name == "Random Forest":
        st.subheader("Feature Importance")
        importances = model.feature_importances_
        feature_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(feature_df.set_index("Feature"))

    # Download model
    st.subheader("Download Trained Model")
    pickle_data = pickle.dumps(model)
    st.download_button(
        label="Download Model (.pkl)",
        data=pickle_data,
        file_name="iris_model.pkl"
    )

# --------------------------------------------------
# TAB 2 — Prediction
# --------------------------------------------------
with tab2:

    st.subheader("Manual Prediction")

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.number_input("Sepal Length", 4.0, 8.0, 5.1)
        sepal_width = st.number_input("Sepal Width", 2.0, 5.0, 3.5)

    with col2:
        petal_length = st.number_input("Petal Length", 1.0, 7.0, 1.4)
        petal_width = st.number_input("Petal Width", 0.1, 3.0, 0.2)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button("Predict"):
        prediction = model.predict(input_data)
        predicted_class = class_names[prediction][0]

        st.success(f"Predicted Class: {predicted_class}")

# --------------------------------------------------
# TAB 3 — Dataset View
# --------------------------------------------------
with tab3:
    st.subheader("Dataset Overview")
    st.dataframe(X.head())

    st.write("Shape:", X.shape)
    st.write("Class Distribution:")
    st.write(pd.Series(y).value_counts())
