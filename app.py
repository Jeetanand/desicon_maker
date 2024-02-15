import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

def preprocess_data(df, outlier_threshold=3):
    # Detect and remove outliers using the Z-score method
    z_scores = (df - df.mean()) / df.std()
    outliers = (z_scores.abs() > outlier_threshold).any(axis=1)
    df_no_outliers = df[~outliers]

    # Delete rows with missing values
    df_no_missing = df_no_outliers.dropna()

    return df_no_missing

def classification(uploaded_df):
    columns=uploaded_df.columns
    X= st.multiselect("Kindly Select the X variables:" , columns)
    if X:
        st.success(f"You have slected following X variable{X}")
        Y= st.selectbox("Kindly Select Y variable:", list(set(columns)- set(X)))
        if Y:
            st.success(f"You have selected {Y} variable")


            # Ensure Y variable is numeric
            if uploaded_df[Y].dtype == 'object':
                label_encoder = LabelEncoder()
                uploaded_df[Y] = label_encoder.fit_transform(uploaded_df[Y])

            X_train, X_test, y_train, y_test = train_test_split(uploaded_df[X], uploaded_df[Y], test_size=0.2, random_state=42)
            # Define classification models
            models = {
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'Support Vector Machine': SVC()
            }


            for model_name, model in models.items():
                st.write(f'### {model_name}')
                model.fit(X_train,y_train)   
                y_pred=model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy:.2f}")
    
                cross_val_scores = cross_val_score(model, uploaded_df[X], uploaded_df[Y], cv=5)
                st.write(f"Cross-validation scores: {cross_val_scores}")
    
                classification_rep = classification_report(y_test, y_pred)
                st.text("Classification Report:")
                st.text(classification_rep)

def regression(uploaded_df):
    columns = uploaded_df.columns
    X = st.multiselect("Kindly Select the X variables:", columns)
    if X:
        st.success(f"You have selected the following X variables: {X}")
        Y = st.selectbox("Kindly Select Y variable:", list(set(columns) - set(X)))
        if Y:
            st.success(f"You have selected {Y} variable")

            # Ensure Y variable is numeric
            if uploaded_df[Y].dtype == 'object':
                label_encoder = LabelEncoder()
                uploaded_df[Y] = label_encoder.fit_transform(uploaded_df[Y])

            X_train, X_test, y_train, y_test = train_test_split(uploaded_df[X], uploaded_df[Y], test_size=0.2, random_state=42)
            
            # Define regression models
            models = {
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Support Vector Machine': SVR()
            }

            for model_name, model in models.items():
                st.write(f'### {model_name}')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                st.write(f"Mean Squared Error: {mse:.2f}")

                cross_val_scores = cross_val_score(model, uploaded_df[X], uploaded_df[Y], cv=5, scoring='neg_mean_squared_error')
                st.write(f"Cross-validation scores: {cross_val_scores}")
            
            



def main():
    st.header("Webapp for making desicon easier and simpler")
    st.subheader("Welcome! 	:wave: This app is designed by Abhijeet Anand. It will help you to make prediction through csv file easier.")
    uploaded_file= st.file_uploader("Choose a CSV file", type=['csv'])
    if uploaded_file:
        st.success("File uploaded sucessfully!")
        uploaded_df= pd.read_csv(uploaded_file)
        st.write(uploaded_df.head(10))

        job = st.selectbox('Select the task:', ['Classification', 'Regression'])
        if job:
            st.success(f"You have seleted {job} Sucessfully!" )
            outlier_threshold = st.slider("Select Outlier Detection Threshold:", min_value=1, max_value=10, value=3)
            uploaded_df = preprocess_data(uploaded_df,outlier_threshold)
            if job=='Classification':
               
                classification(uploaded_df)
            elif job=='Regression':
                
               
                regression(uploaded_df)


    




if __name__ == "__main__":
    main()