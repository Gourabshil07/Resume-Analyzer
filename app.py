import streamlit as st
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import re
import os


if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY missing in Streamlit Secrets!")
    st.stop()

# Load environment variables
api_key = st.secrets["GROQ_API_KEY"]


# Session states
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "resume" not in st.session_state:
    st.session_state.resume = ""
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""

st.title("Resume Analyzer 📝")

# Function to extract text from PDF
def extract_pdf_text(uploaded_file):
    try:
        return extract_text(uploaded_file)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

# Function to calculate similarity
def calculate_similarity_bert(text1, text2):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings1 = model.encode([text1])
    embeddings2 = model.encode([text2])
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return similarity

# Function to get report from Groq API
def get_report(resume, job_desc):
    client = Groq(api_key=api_key)
    prompt = f"""
    You are an ATS evaluator and career expert. 
    Analyze the candidate's resume based on the job description below.

    

    Candidate Resume:
    {resume}

    Job Description:
    {job_desc}

    Output:
    - List each missing skills and  requirement skills from the job description.
    - End with "Suggestions to improve your resume:" with detailed tips and add some youtube videos links based on the missing skills 
     where user learn the skills and imporves the knowledge( use only high-quality tech channels like Kunal Kushwaha, CodeWithHarry, FreeCodeCamp, Simplilearn, etc).
    """
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

# Extract scores from report
def extract_scores(text):
    pattern = r'(\d+(?:\.\d+)?)/5'
    matches = re.findall(pattern, text)
    return [float(match) for match in matches]

# ---------- Streamlit UI Flow ----------

if not st.session_state.form_submitted:
    with st.form("my_form"):
        resume_file = st.file_uploader("Upload your Resume/CV (PDF format)", type="pdf")
        st.session_state.job_desc = st.text_area("Enter the Job Description:", placeholder="Paste the job description here...")
        submitted = st.form_submit_button("Analyze")

        if submitted:
            if st.session_state.job_desc and resume_file:
                st.info("Extracting information...")
                st.session_state.resume = extract_pdf_text(resume_file)
                st.session_state.form_submitted = True
                st.rerun()
            else:
                st.warning("Please upload both Resume and Job Description.")

if st.session_state.form_submitted:
    

    report = get_report(st.session_state.resume, st.session_state.job_desc)

    # Calculate similarity score (0 to 1)
    similarity_score = calculate_similarity_bert(st.session_state.resume, st.session_state.job_desc)

# Convert to percentage
    match_percentage = round(similarity_score * 100, 2)

    st.subheader("Resume Match Score:")
    match_percentage = float(match_percentage)
    match_percentage = round(similarity_score * 100, 2)

    st.write(f"**Match Percentage:** {match_percentage}%")

    report_scores = extract_scores(report)
    avg_score = sum(report_scores) / len(report_scores) if report_scores else 0


    st.subheader("Resume Analysis Report:")
    st.write("### 📄 Detailed Resume Report:")

    # Render the full report safely on all devices
    st.text_area("Report:", report, height=400)

    st.download_button("Download Report", data=report, file_name="resume_report.txt", icon=":material/download:")

    if st.button("🔄 Upload Another Resume"):
        st.session_state.form_submitted = False
        st.session_state.resume = ""
        st.session_state.job_desc = ""
        st.rerun()
