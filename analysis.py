from pathlib import Path

import numpy as np
import pandas as pd


def load_and_prepare_data(filepath):
    # Load the CSV file into a Pandas DataFrame
    data_path = Path(filepath)
    if not data_path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        candidates = (
            Path.cwd() / data_path,
            project_root / data_path,
            project_root / "Data" / data_path.name,
        )
        data_path = next((path for path in candidates if path.exists()), candidates[0])

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. "
            "Place Maths.csv in the project's data folder."
        )

    df = pd.read_csv(data_path)

    # Create Result based on the final grade (G3)
    # G3 = 0 is treated as Dropout, not as a zero score
    df["Result"] = df["G3"].apply(
        lambda x: "Dropout" if x == 0 else ("Fail" if 1 <= x <= 9 else "Pass")
    )

    # Convert G3 (out of 20) into a percentage
    df["Percentage"] = (df["G3"] / 20) * 100

    # Calculate average alcohol consumption from weekday and weekend values
    df["avg_alcohol"] = (df["Dalc"] + df["Walc"]) / 2

    # Calculate average education level of both parents
    df["parent_edu_avg"] = (df["Medu"] + df["Fedu"]) / 2

    # Calculate the grade trend from the first grade to the final grade
    df["grade_trend"] = df["G3"] - df["G1"]

    # Count how many types of support are marked as "yes"
    df["total_support"] = (
        (df["schoolsup"] == "yes").astype(int)
        + (df["famsup"] == "yes").astype(int)
        + (df["paid"] == "yes").astype(int)
    )

    # Calculate the student's academic risk score
    df["risk_score"] = (
        (df["failures"] * 2)
        + (df["absences"] / 10)
        + df["avg_alcohol"]
        - df["studytime"]
    )

    # Calculate the average of G1 and G2
    df["g1_g2_avg"] = (df["G1"] + df["G2"]) / 2

    # Return the complete prepared DataFrame
    return df

def calculate_statistics(df):
    # Exclude students with G3 = 0 (dropouts) for academic performance statistics
    non_dropout = df[df["G3"] != 0]

    # Calculate the average final grade (G3) among non-dropout students
    class_avg_g3 = np.mean(non_dropout["G3"])

    # Calculate the number of students who passed (G3 >= 10)
    pass_count = np.sum(non_dropout["G3"] >= 10)

    # Calculate pass rate as a percentage of non-dropout students
    pass_rate = (pass_count / len(non_dropout)) * 100

    # Count students whose G3 is exactly 0 (dropouts)
    dropout_count = np.sum(df["G3"] == 0)

    # Count students whose G3 is between 1 and 9 (at-risk/failing)
    at_risk_count = np.sum((df["G3"] >= 1) & (df["G3"] <= 9))

    # Calculate the correlation matrix of G1, G2, and G3
    # Only non-dropout students are included
    correlation_matrix = np.corrcoef(
        non_dropout[["G1", "G2", "G3"]].values,
        rowvar=False
    )

    # Return all calculated statistics as a dictionary
    return {
        "class_avg_g3": class_avg_g3,
        "pass_rate": pass_rate,
        "dropout_count": dropout_count,
        "at_risk_count": at_risk_count,
        "correlation_matrix": correlation_matrix
    }
def create_visualizations(df):
    # Import required libraries
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    # Create the output folder if it does not already exist
    os.makedirs("output", exist_ok=True)

    # -------------------------------
    # Chart 1: Average G3 by Study Time
    # -------------------------------

    # Calculate average G3 for each studytime level (1, 2, 3, 4)
    avg_g3 = df.groupby("studytime")["G3"].mean()

    # Define studytime levels
    studytime_levels = [1, 2, 3, 4]

    # Get the corresponding average G3 values
    avg_values = [avg_g3.get(level, np.nan) for level in studytime_levels]

    # Create the bar chart
    plt.figure()
    plt.bar(studytime_levels, avg_values)

    # Add chart title and axis labels
    plt.title("Average G3 by Study Time")
    plt.xlabel("Study Time (1=<2hrs, 2=2-5hrs, 3=5-10hrs, 4=>10hrs)")
    plt.ylabel("Average G3")

    # Ensure all studytime levels appear on the X axis
    plt.xticks(studytime_levels)

    # Save the chart
    plt.savefig("output/avg_g3_by_studytime.png")

    # Close the chart to free memory
    plt.close()

    # -------------------------------
    # Chart 2: Student Result Distribution
    # -------------------------------

    # Count the number of students in each Result category
    result_counts = df["Result"].value_counts()

    # Ensure the slices appear in the requested order
    labels = ["Pass", "Fail", "Dropout"]
    sizes = [result_counts.get(label, 0) for label in labels]

    # Create the pie chart
    plt.figure()
    plt.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%"
    )

    # Add chart title
    plt.title("Student Result Distribution")

    # Save the pie chart
    plt.savefig("output/pass_fail_dropout_pie.png")

    # Close the chart to free memory
    plt.close()
def generate_interactive_charts(df):
    # Import Plotly Express for creating interactive charts
    import plotly.express as px

    # --------------------------------
    # Chart 1: Study Time vs Final Grade
    # --------------------------------

    # Create an interactive scatter plot
    fig1 = px.scatter(
        df,
        x="studytime",
        y="G3",
        color="Result",
        hover_data=["absences", "G1", "G2"],
        title="Study Time vs Final Grade (G3)",
        color_discrete_map={
            "Pass": "green",
            "Fail": "red",
            "Dropout": "grey"
        }
    )

    # Display the interactive scatter plot
    fig1.show()

    # --------------------------------
    # Chart 2: Average G3 by Internet Access
    # --------------------------------

    # Calculate average G3 for each internet access group
    avg_g3_internet = (
        df.groupby("internet", as_index=False)["G3"]
        .mean()
    )

    # Create an interactive bar chart
    fig2 = px.bar(
        avg_g3_internet,
        x="internet",
        y="G3",
        color="internet",
        title="Average G3 by Internet Access"
    )

    # Display the interactive bar chart
    fig2.show()


def print_summary(stats):
    # Print the formatted header
    print("=" * 48)
    print("STUDENT ACADEMIC RISK INTELLIGENCE SYSTEM")
    print("ANALYSIS SUMMARY")
    print("=" * 48)

    # Print each statistic in a clean format
    print(f"Total Students        : {stats['dropout_count'] + stats['at_risk_count']}")
    print(f"Class Average G3      : {stats['class_avg_g3']:.2f}")
    print(f"Pass Rate             : {stats['pass_rate']:.2f}%")
    print(f"At-Risk Count         : {stats['at_risk_count']}")
    print(f"Dropout Count         : {stats['dropout_count']}")

    # Print the closing separator
    print("=" * 48)


# Main program block
if __name__ == "__main__":
    # Load and prepare the student dataset
    df = load_and_prepare_data("data/Maths.csv")

    # Calculate academic statistics
    stats = calculate_statistics(df)

    # Generate and save static charts
    create_visualizations(df)

    # Generate and display interactive Plotly charts
    generate_interactive_charts(df)

    # Print the analysis summary
    print_summary(stats)

    # Confirm that the analysis has completed
    print("Analysis complete. Charts saved to output/ folder")
    