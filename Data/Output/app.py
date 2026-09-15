import streamlit as st
import pandas as pd
import plotly.express as px


# Configure the Streamlit page
st.set_page_config(
    page_title="Student Academic Risk Intelligence System",
    layout="wide"
)


# Load the Maths.csv dataset
df = pd.read_csv("data/Maths.csv")


# Apply the same feature engineering used in analysis.py

# Create Result based on G3
# G3 = 0: Dropout, 1-9: Fail, 10-20: Pass
def get_result(g3):
    if g3 == 0:
        return "Dropout"
    elif 1 <= g3 <= 9:
        return "Fail"
    else:
        return "Pass"


df["Result"] = df["G3"].apply(get_result)

# Convert G3 into percentage
df["Percentage"] = (df["G3"] / 20) * 100

# Calculate average alcohol consumption
df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

# Calculate average education level of both parents
df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

# Calculate grade trend from G1 to G3
df["grade_trend"] = df["G3"] - df["G1"]

# Count "yes" values across support-related columns
support_columns = ["schoolsup", "famsup", "paid"]
df["total_support"] = (
    df[support_columns]
    .apply(lambda row: (row == "yes").sum(), axis=1)
)

# Calculate academic risk score
df["risk_score"] = (
    (df["failures"] * 2)
    + (df["absences"] / 10)
    + df["avg_alcohol"]
    - df["studytime"]
)

# Calculate average of G1 and G2
df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2


# Display the main dashboard title
st.title("🎓 Student Academic Risk Intelligence System")


# Prepare non-dropout students for class average and pass rate
non_dropout = df[df["G3"] != 0]

# Calculate class average G3 excluding dropouts
class_average_g3 = non_dropout["G3"].mean()

# Calculate pass rate among non-dropout students
pass_count = (non_dropout["G3"] >= 10).sum()
pass_rate = (pass_count / len(non_dropout)) * 100

# Calculate at-risk student count
at_risk_count = ((df["G3"] >= 1) & (df["G3"] <= 9)).sum()


# Create four KPI cards in one row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Students", len(df))

with col2:
    st.metric("Class Average G3", f"{class_average_g3:.2f}")

with col3:
    st.metric("Pass Rate %", f"{pass_rate:.1f}%")

with col4:
    st.metric("At-Risk Count", int(at_risk_count))

    # Performance Charts section
st.subheader("📊 Performance Charts")

# Create two columns so the charts appear side by side
col1, col2 = st.columns(2)

# -------------------------------
# Left Chart: Study Time vs Final Grade
# -------------------------------
with col1:
    # Define colors for each student result category
    result_colors = {
        "Pass": "green",
        "Fail": "red",
        "Dropout": "grey"
    }

    # Create the interactive scatter plot
    fig1 = px.scatter(
        df,
        x="studytime",
        y="G3",
        color="Result",
        color_discrete_map=result_colors,
        hover_data=["absences", "G1", "G2"],
        title="Study Time vs Final Grade"
    )

    # Display the scatter plot
    st.plotly_chart(fig1, use_container_width=True)


# -------------------------------
# Right Chart: Average G3 by Internet Access
# -------------------------------
with col2:
    # Calculate average G3 for each internet access group
    avg_g3_internet = (
        df.groupby("internet", as_index=False)["G3"]
        .mean()
    )

    # Create the interactive bar chart
    fig2 = px.bar(
        avg_g3_internet,
        x="internet",
        y="G3",
        color="internet",
        title="Average G3 by Internet Access"
    )

    # Display the bar chart
    st.plotly_chart(fig2, use_container_width=True)