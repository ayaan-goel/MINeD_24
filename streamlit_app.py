import streamlit as st
import pdfplumber
import docx2txt  # Library for extracting text from DOCX files
from bs4 import BeautifulSoup  # Library for parsing HTML
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Set page configuration
st.set_page_config(page_title="ResuMAGIC", page_icon="🌟", layout="wide")

# Initialize Gemini LLM with Google API key
google_api_key = "AIzaSyDX7iqE8XTN8npHp9jKZST8HMfZS4ncNpg"  # Replace with your Google API key
llm = GoogleGenerativeAI(temperature=0.1, google_api_key=google_api_key, model="gemini-pro")

# Define prompt templates using f-strings
first_input_prompt_template = "Please provide a rewritten version of {text}."
second_input_prompt_template = "Please extract and provide education details from {descript}."
work_prompt_template = "Please extract and provide work experience details from {text}."
projects_prompt_template = "Please extract and provide project details from {text}."
skills_prompt_template = "Please extract and provide skills from {text}."
career_trajectory_prompt_template = "Based on the provided education, work experience, and projects, analyze the career trajectory also mention year if given."

# Initialize Prompt Templates
first_input_prompt = PromptTemplate(
    input_variables=['text'],
    template=first_input_prompt_template
)
second_input_prompt = PromptTemplate(
    input_variables=['descript'],
    template=second_input_prompt_template
)
work_prompt = PromptTemplate(
    input_variables=['text'],
    template=work_prompt_template
)
projects_prompt = PromptTemplate(
    input_variables=['text'],
    template=projects_prompt_template
)
skills_prompt = PromptTemplate(
    input_variables=['text'],
    template=skills_prompt_template
)
career_trajectory_prompt = PromptTemplate(
    input_variables=['education', 'work', 'projects'],
    template=career_trajectory_prompt_template
)

# Chain of LLMs
chain1 = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='descript')
chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='descript_two')

# Initialize Chains for Work, Projects, Skills, and Career Trajectory
work_chain = LLMChain(llm=llm, prompt=work_prompt, verbose=True, output_key='work_details')
projects_chain = LLMChain(llm=llm, prompt=projects_prompt, verbose=True, output_key='projects_details')
skills_chain = LLMChain(llm=llm, prompt=skills_prompt, verbose=True, output_key='skills_details')
career_trajectory_chain = LLMChain(llm=llm, prompt=career_trajectory_prompt, verbose=True, output_key='career_trajectory')

# Parent Chain for all tasks
parent_chain = SequentialChain(chains=[chain1, chain2, work_chain, projects_chain, skills_chain, career_trajectory_chain], input_variables=['text'], output_variables=['descript_two', 'work_details', 'projects_details', 'skills_details', 'career_trajectory'], verbose=True)

# Streamlit UI
st.title('ResuMAGIC AI 🌟')

# File uploader for resume PDF, DOCX, and HTML
uploaded_file = st.file_uploader("Upload Resume PDF, DOCX, or HTML", type=['pdf', 'docx', 'html'])

if uploaded_file is not None:
    # Display loading spinner while processing the file
    with st.spinner("Analyzing..."):
        # Extract text from uploaded file
        if uploaded_file.type == 'application/pdf':
            # Extract text from PDF
            with pdfplumber.open(uploaded_file) as pdf:
                extracted_text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    extracted_text += page_text + "\n"
        elif uploaded_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            # Extract text from DOCX
            extracted_text = docx2txt.process(uploaded_file)
        elif uploaded_file.type == 'text/html':
            # Extract text from HTML
            soup = BeautifulSoup(uploaded_file, 'html.parser')
            extracted_text = soup.get_text()

        # Display extracted text
        st.subheader("Check Out the Outcomes :")

        # Execute the parent chain using the extracted text
        if extracted_text:
            try:
                result = parent_chain({'text': extracted_text})

                
                st.write(result['descript_two'])

                st.write(result['work_details'])

                st.write(result['projects_details'])

                st.write(result['skills_details'])

                st.write(result['career_trajectory'])

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
else:
    st.info("Please upload a PDF, DOCX, or HTML file to analyze.")
