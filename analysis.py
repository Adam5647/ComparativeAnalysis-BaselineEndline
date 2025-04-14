import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import io
import requests
import json
from sklearn.cluster import KMeans
from fpdf import FPDF
import base64

# --- Correct Answers Dictionary ---
CORRECT_ANSWERS = {
    "1": ["73rd Amendment Act"],
    "3": [
        "General Standing Committee",
        "Finance, Audit and Planning Committee",
        "Social Justice Committee"
    ],
    "4": ["15 days notice must be given for the Grama Sabha"]
}

# --- Helper Functions ---
def extract_question_number(question_text):
    return question_text.strip().split('.')[0]

def clean_response(response):
    if pd.isna(response):
        return []
    return [opt.strip().split('. ', 1)[-1] for opt in str(response).split(',') if opt.strip()]

def score_response(q_num, response):
    correct = CORRECT_ANSWERS[q_num]
    if q_num in ["1", "4"]:
        return 100.0 if response and response[0] in correct else 0.0
    elif q_num == "3":
        correct_set = set(correct)
        selected_set = set(response)
        score = len(correct_set & selected_set) / len(correct_set) * 100
        return round(score, 2)
    return 0.0

def preprocess(df):
    df["Q_num"] = df["Question"].apply(extract_question_number)
    df["Cleaned_Response"] = df["Responses"].apply(clean_response)
    df["Score"] = df.apply(lambda row: score_response(row["Q_num"], row["Cleaned_Response"]), axis=1)
    pivot_scores = df.pivot_table(index="Name", columns="Q_num", values="Score", aggfunc="first")
    pivot_scores["Total"] = pivot_scores.mean(axis=1).round(2)
    return df, pivot_scores.reset_index()

def generate_pdf_report(df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Baseline vs Endline Report", ln=True, align='C')
    pdf.ln(10)
    for i, row in df.iterrows():
        pdf.cell(200, 10, txt=f"{row['Name']}: {row['Total_Baseline']}% -> {row['Total_Endline']}% | Change: {row['Improvement']:.2f}%", ln=True)
    
    pdf_binary = pdf.output(dest='S').encode('latin1')
    return pdf_binary

def get_pdf_download_link(binary_pdf):
    b64 = base64.b64encode(binary_pdf).decode('utf-8')
    return f'<a href="data:application/pdf;base64,{b64}" download="report.pdf">Download PDF Report</a>'

# --- Load Data ---
baseline = pd.read_excel("Baseline.xlsx")
endline = pd.read_excel("Endline.xlsx")

baseline_long, baseline_scores = preprocess(baseline)
endline_long, endline_scores = preprocess(endline)

# Merge for comparison
comparison = pd.merge(baseline_scores, endline_scores, on="Name", how="inner", suffixes=("_Baseline", "_Endline"))
comparison["Improvement"] = comparison["Total_Endline"] - comparison["Total_Baseline"]
comparison["Status"] = comparison["Improvement"].apply(lambda x: "Improved" if x > 0 else ("No Change" if x == 0 else "Declined"))

# Clustering Participants
features = comparison[["Total_Baseline", "Total_Endline", "Improvement"]]
kmeans = KMeans(n_clusters=3, random_state=0).fit(features)
comparison["Cluster"] = kmeans.labels_

# Merge full long data
baseline_long["Phase"] = "Baseline"
endline_long["Phase"] = "Endline"
full_long = pd.concat([baseline_long, endline_long], ignore_index=True)

# --- Streamlit App ---
st.set_page_config(page_title="Baseline vs Endline Dashboard", layout="wide")
st.title("📊 Baseline vs Endline Comparative Analysis")

# Session Tracking
if "session_count" not in st.session_state:
    st.session_state.session_count = 0
st.session_state.session_count += 1
st.sidebar.success(f"Session: {st.session_state.session_count}")

# --- Tabs ---
tabs = st.tabs(["Overview", "Participant-wise Analysis", "AI Insights - Summary", "AI Insights - Outliers & Trends"])

with tabs[0]:
    st.subheader("📋 Global Participant Response Table")
    pivoted = full_long.pivot_table(index=["Name", "Q_num"], columns="Phase", values=["Responses", "Score"], aggfunc="first")
    pivoted.columns = [f"{i}_{j}" for i, j in pivoted.columns]
    pivoted.reset_index(inplace=True)
    st.dataframe(pivoted)

    st.download_button(
        label="📥 Download Global Table as CSV",
        data=pivoted.to_csv(index=False).encode("utf-8"),
        file_name="participant_response_comparison.csv",
        mime="text/csv"
    )

    st.subheader("📈 Individual Scores Summary")
    st.dataframe(comparison)

    if st.button("📄 Generate PDF Report"):
        pdf_binary = generate_pdf_report(comparison)
        st.markdown(get_pdf_download_link(pdf_binary), unsafe_allow_html=True)
        
    st.subheader("📊 Total Score Comparison")

    bar_fig = px.bar(
        comparison.melt(id_vars="Name", value_vars=["Total_Baseline", "Total_Endline"], 
                        var_name="Phase", value_name="Total Score"),
        x="Name",
        y="Total Score",
        color="Phase",
        barmode="group",
        title="Participant Total Scores: Baseline vs Endline"
    )
    bar_fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(bar_fig, use_container_width=True)

    st.subheader("🧪 Clustering Insights")
    cluster_fig = px.scatter(
        comparison,
        x="Total_Baseline",
        y="Total_Endline",
        color="Cluster",
        hover_name="Name",
        size=comparison["Improvement"].clip(lower=0.01)
    )
    st.plotly_chart(cluster_fig, use_container_width=True)

with tabs[1]:
    st.subheader("🧑‍💼 Participant-wise Analysis")
    participant_names = full_long["Name"].unique()
    selected_name = st.selectbox("Select a Participant", sorted(participant_names))

    selected_data = full_long[full_long["Name"] == selected_name]
    st.markdown(f"### Responses of {selected_name}")
    st.dataframe(selected_data[["Phase", "Question", "Responses", "Score"]])

    fig2, ax2 = plt.subplots()
    sns.barplot(data=selected_data, x="Q_num", y="Score", hue="Phase", ax=ax2)
    ax2.set_title(f"{selected_name} - Score by Question")
    st.pyplot(fig2)

with tabs[2]:
    st.subheader("🤖 AI Insights - Summary Level")
    user_prompt = st.text_area("Optional: Add your custom context or prompt for the AI")
    if st.button("Generate Summary Insights"):
        top_improvers = comparison.nlargest(3, "Improvement")
        top_decliners = comparison.nsmallest(3, "Improvement")
        baseline_avg = comparison["Total_Baseline"].mean()
        endline_avg = comparison["Total_Endline"].mean()
        avg_improvement = comparison["Improvement"].mean()

        base_prompt = f"""
You are an education data analyst.

You are given performance data comparing baseline and endline assessments. Provide an analytical summary:

### Executive Summary:
- Total improved: {comparison[comparison['Status']=='Improved'].shape[0]}
- Average improvement: {avg_improvement:.2f}%
- Average Baseline Score: {baseline_avg:.2f}%
- Average Endline Score: {endline_avg:.2f}%

### Top 3 Improvers:
{top_improvers[['Name', 'Improvement']].to_string(index=False)}

### Top 3 Decliners:
{top_decliners[['Name', 'Improvement']].to_string(index=False)}

### Table Summary:
{comparison[['Name', 'Total_Baseline', 'Total_Endline', 'Improvement']].to_markdown(index=False)}
"""

        if user_prompt.strip():
            base_prompt += f"\n\nAdditional Notes: {user_prompt.strip()}"

        try:
            response = requests.post(
                "https://33aa-2405-201-ac0b-e0cb-f813-6255-431f-b370.ngrok-free.app/api/generate",
                json={"model": "mistral", "prompt": base_prompt, "stream": False},
                headers={"Content-Type": "application/json"}
            )
            result = response.json()
            st.markdown("### AI Generated Insights")
            st.write(result.get("response", "No response from model."))
        except Exception as e:
            st.error(f"Error fetching insights from AI model: {e}")

with tabs[3]:
    st.subheader("🤖 AI Insights - Outliers and Performance Patterns")
    user_prompt2 = st.text_area("Optional: Add custom context for performance pattern insights")
    if st.button("Generate Advanced Insights"):
        participant_list = "\n".join(comparison['Name'].tolist())
        simplified_table = comparison[['Name', 'Total_Baseline', 'Total_Endline', 'Improvement']].to_markdown(index=False)

        base_advanced = f'''
You are a senior education performance analyst.

You are analyzing participant scores from two sessions: baseline and endline.
Only analyze the participants listed below. DO NOT add fictional names.

Participants:
{participant_list}

Using the following table, provide:
- ✅ Top 3 participants with highest improvement
- 📉 Top 3 with largest decline
- 📊 Cluster patterns or trends
- ❗ Which questions were weakest or strongest overall
- 🔄 Recommendations for each cluster if possible
'''
        if user_prompt2.strip():
            base_advanced += f"\nUser prompt: {user_prompt2.strip()}"

        base_advanced += f"\nHere is the data:\n{simplified_table}"

        try:
            advanced_response = requests.post(
                "https://33aa-2405-201-ac0b-e0cb-f813-6255-431f-b370.ngrok-free.app/api/generate",
                json={"model": "mistral", "prompt": base_advanced, "stream": False},
                headers={"Content-Type": "application/json"}
            )
            insights = advanced_response.json()
            st.markdown("### AI Advanced Analysis")
            st.write(insights.get("response", "No insights returned by model."))
        except Exception as e:
            st.error(f"Error during advanced AI insight generation: {e}")
