import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# Title
st.title("Fake News Detection")

# Load Dataset
try:
    fake_path = "Fake.csv"  # Path to Fake.csv in your project directory
    true_path = "True.csv"  # Path to True.csv in your project directory

    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)

    # Adding target labels
    df_fake["class"] = 0
    df_true["class"] = 1

    # Combine datasets
    data = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)

    # Sidebar options for data visualization
    st.sidebar.header("Options")
    view_data = st.sidebar.checkbox("View Dataset")
    view_stats = st.sidebar.checkbox("View Dataset Statistics")

    if view_data:
        st.subheader("Dataset Preview")
        st.write(data.sample(10))  # Display a random sample of 10 rows

    if view_stats:
        st.subheader("Dataset Statistics")
        st.write(data.describe(include="all"))  # Display dataset statistics
        st.write("Class Distribution:")
        st.bar_chart(data["class"].value_counts())

    # Split data into training and test sets
    X = data['text']
    y = data['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Text vectorization
    vectorizer = CountVectorizer(stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Test on user input
    user_input = st.text_area("Enter news text to predict:")
    if st.button("Predict"):
        if user_input.strip():
            input_vec = vectorizer.transform([user_input])
            prediction = model.predict(input_vec)
            st.write("Prediction: ", "REAL" if prediction[0] == 1 else "FAKE")
        else:
            st.error("Please enter some text to predict.")

except FileNotFoundError as e:
    st.error(f"Dataset not found: {e}")
