# =========================
# IMPORT LIBRARIES
# =========================
import streamlit as st
import joblib
import pandas as pd

# =========================
# LOAD FILES
# =========================
try:
    model = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")

except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

# =========================
# HELPER FUNCTION
# =========================
def get_options(prefix):
    opts = [col.replace(prefix, "") for col in columns if col.startswith(prefix)]
    return sorted(list(set(opts)))

# =========================
# OPTIONS
# =========================
job_options = ["Other"] + get_options("job_title_")
edu_options = ["Other"] + get_options("education_level_")
loc_options = ["Other"] + get_options("location_")
ind_options = ["Other"] + get_options("industry_")
company_options = ["Other"] + get_options("company_size_")
remote_options = ["Other"] + get_options("remote_work_")

# =========================
# TITLE
# =========================
st.title("💼 Salary Prediction App")

# =========================
# USER INPUT
# =========================
exp = st.number_input("Experience (years)", min_value=0, max_value=30, value=1)

skills = st.number_input("Skills Count", min_value=0, max_value=50, value=1)

cert = st.number_input("Certifications", min_value=0, max_value=20, value=0)

job = st.selectbox("Job Role", job_options)
edu = st.selectbox("Education", edu_options)
loc = st.selectbox("Location", loc_options)
ind = st.selectbox("Industry", ind_options)
company = st.selectbox("Company Size", company_options)
remote = st.selectbox("Remote Work", remote_options)

# =========================
# CREATE INPUT DATAFRAME
# =========================
input_dict = {
    "experience_years": exp,
    "skills_count": skills,
    "certifications": cert,
    "job_title": job,
    "education_level": edu,
    "location": loc,
    "industry": ind,
    "company_size": company,
    "remote_work": remote
}

input_df = pd.DataFrame([input_dict])

# =========================
# FEATURE ENGINEERING
# =========================
input_df['exp_squared'] = input_df['experience_years'] ** 2

input_df['skill_per_exp'] = (
    input_df['skills_count'] /
    (input_df['experience_years'] + 1)
)

input_df['cert_per_skill'] = (
    input_df['certifications'] /
    (input_df['skills_count'] + 1)
)

input_df['seniority'] = pd.cut(
    input_df['experience_years'],
    bins=[-1, 2, 5, 10, 100],
    labels=['Fresher', 'Junior', 'Mid', 'Senior']
)

# =========================
# CONVERT TO DUMMIES
# =========================
input_df = pd.get_dummies(input_df)

# =========================
# ALIGN COLUMNS
# =========================
input_df = input_df.reindex(columns=columns, fill_value=0)

# =========================
# SCALE NUMERIC FEATURES
# =========================
num_cols = [
    'experience_years',
    'skills_count',
    'certifications',
    'exp_squared',
    'skill_per_exp',
    'cert_per_skill'
]

input_df[num_cols] = scaler.transform(input_df[num_cols])

# =========================
# PREDICT
# =========================
if st.button("Predict Salary"):

    prediction = model.predict(input_df)

    st.success(
        f"💰 Predicted Salary: ₹ {int(prediction[0]):,}"
    )

    st.balloons()
