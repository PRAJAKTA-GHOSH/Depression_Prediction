import streamlit as st
import pickle
import pandas as pd

# Loading the model 
with open('model_decision_tree.pkl', "rb") as file:
    model, columns = pickle.load(file)

def preprocess_input(days, gender, age, afftype, melanch, inpatient, edu, marriage, work, madrs1, madrs2):
    # Converting age to average of range as in the dataset
    age = (int(age.split('-')[0]) + int(age.split('-')[1])) / 2
    
    # Mapping the education(edu) to numeric values
    edu_dict = {"6-10": 8, "11-15": 13, "16-20": 18}
    edu = edu_dict[edu]
    
    # Preparing the input data 
    input_data = {
        'days': days,
        'madrs1': madrs1,
        'madrs2': madrs2,
        'age': age,
        'gender_2': 1 if gender == '2' else 0,
        'afftype_2': 1 if afftype == '2' else 0,
        'afftype_3': 1 if afftype == '3' else 0,
        'melanch_2': 1 if melanch == '2' else 0,
        'inpatient_2': 1 if inpatient == '2' else 0,
        'edu_8': 1 if edu == 8 else 0,
        'edu_13': 1 if edu == 13 else 0,
        'edu_18': 1 if edu == 18 else 0,
        'marriage_2': 1 if marriage == '2' else 0,
        'marriage_3': 1 if marriage == '3' else 0,
        'work_2': 1 if work == '2' else 0
    }
    
    # Converting the dictionary to DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Adding the  missing columns with zero values
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Matching the training data
    input_df = input_df[columns]
    
    return input_df

def predict(days, gender, age, afftype, melanch, inpatient, edu, marriage, work, madrs1, madrs2):
    input_df = preprocess_input(days, gender, age, afftype, melanch, inpatient, edu, marriage, work, madrs1, madrs2)
    prediction = model.predict(input_df)
    return prediction

def main():
    st.title("DEPRESSION Prediction")

    days = st.text_input("Days", "11")
    gender = st.selectbox("Gender", ["1", "2"])
    age = st.selectbox("Age", ["20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69"])
    afftype = st.selectbox("Afftype", ["1", "2", "3"])
    melanch = st.selectbox("Melanch", ["1", "2"])
    inpatient = st.selectbox("Inpatient", ["1", "2"])
    edu = st.selectbox("Edu", ["6-10", "11-15", "16-20"])  
    marriage = st.selectbox("Marriage", ["1", "2", "3"])
    work = st.selectbox("Work", ["1", "2"])
    madrs1 = st.text_input("Madrs1", "19")
    madrs2 = st.text_input("Madrs2", "19")
    
    result = ""
    
    if st.button("Predict"):
        try:
            # Converting input text to data types required
            days = float(days)
            madrs1 = float(madrs1)
            madrs2 = float(madrs2)
            
            # Making the prediction
            result = predict(days, gender, age, afftype, melanch, inpatient, edu, marriage, work, madrs1, madrs2)
            st.success(f"The predicted depression level is {result[0]}")
        except ValueError as e:
            st.error(f"Error: {str(e)}. Please enter valid numbers and ensure all fields are filled.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
