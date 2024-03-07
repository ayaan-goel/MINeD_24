import pandas as pd
import streamlit as st
import pdfplumber
import docx2txt  # Library for extracting text from DOCX files
from bs4 import BeautifulSoup  # Library for parsing HTML
from PIL import Image
import easyocr 
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Set page configuration
st.set_page_config(page_title="ResuMAGIC", page_icon="🌟", layout="wide")

# Load O*NET database (assuming it's in CSV format)
onet_database = "test_two.csv"
df_onet = pd.read_csv(onet_database)

# Initialize Gemini LLM with Google API key
google_api_key = "AIzaSyAkThsHjqoxUTjLT82GAIe1tqrwMe-GCys"  # Replace with your Google API key
llm = GoogleGenerativeAI(temperature=0.1, google_api_key=google_api_key, model="gemini-pro")

# Define function to map job description to standardized job title
def map_job_description_to_title(job_description):
    # Your mapping logic here
    # For example, find the closest match based on similarity of job descriptions
    similarity_scores = []
    for index, row in df_onet.iterrows():
        similarity_score = your_similarity_function(job_description, row['job_description'])
        similarity_scores.append((row['standerlised_tittle'], similarity_score))
    
    # Sort similarity scores and return the most similar job title
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    return sorted_scores[0][0] if sorted_scores[0][1] > 0 else None

# Function to calculate similarity between job descriptions
def your_similarity_function(job_desc1, job_desc2):
    # Your similarity calculation method here
    # You can use techniques like cosine similarity, Jaccard similarity, etc.
    # For simplicity, let's assume a basic string matching approach for demonstration
    return sum(a == b for a, b in zip(job_desc1.lower(), job_desc2.lower())) / max(len(job_desc1), len(job_desc2))
# Define prompt templates using f-strings
first_input_prompt_template = "Please provide a rewritten version of {text}. If no relevant information is available, please indicate that information is not given."
second_input_prompt_template = "Please extract and provide education details from {descript}. If no relevant information is available, please indicate that information is not given."
work_prompt_template = "Please extract and provide work experience details from {text}. If no relevant information is available, please indicate that information is not given."
projects_prompt_template = "Please extract and provide project details from {text}. If no relevant information is available, please indicate that information is not given."
skills_prompt_template = "Please extract and provide skills from {text}. If no relevant information is available, please indicate that information is not given."
career_trajectory_prompt_template = "Based on the provided {work_details} experience, analyze the career trajectory and give output in the following format: 1. job_title (start_date) >> job_title_2 (start_date) >> job_title_3 (start_date) >> job_title_4 (start_date) >> job_title_5 (start_date) >> job_title_6 (start_date). Please include only the start dates for each job title to analyze the career trajectory. If no relevant information is available, please indicate that information is not given."

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
    input_variables=['text'],
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

# File uploader for resume PDF, DOCX, HTML, and JPEG
uploaded_file = st.file_uploader("Upload Resume PDF, DOCX, HTML, or JPEG", type=['pdf', 'docx', 'html', 'jpeg', 'jpg'])

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
        elif uploaded_file.type.startswith('image/'):
            # Extract text from JPEG using EasyOCR
            reader = easyocr.Reader(['en'])  # Specify language(s) as needed
            result = reader.readtext(uploaded_file.read())
            extracted_text = result    
            

        # Display extracted text
        st.subheader("Check Out the Outcomes :")

        # Execute the parent chain using the extracted text
        if extracted_text:
            try:
                result = parent_chain({'text': extracted_text})

                # Map the extracted text to standardized job titles
                standardized_job_titles = []
                for extracted_job_desc in result.get('descript_two', []):
                    standardized_job_title = map_job_description_to_title(extracted_job_desc)
                    if standardized_job_title:
                        standardized_job_titles.append(standardized_job_title)

               
                st.write(result.get('descript_two',[]))
                st.write(result.get('projects_details', []))
                st.write(result.get('skills_details', []))
                
                st.write("Career Tracjectory:")
                st.write(result.get('career_trajectory', []))

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
else:
    st.info("Please upload a PDF, DOCX, HTML, or JPEG file to analyze.")
