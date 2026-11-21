"""ResuMAGIC - upload a resume, get it back as clean structured data."""

import hashlib
import json
import os

import streamlit as st

import core

st.set_page_config(page_title="ResuMAGIC", page_icon="🌟", layout="wide")


def get_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        try:
            key = st.secrets["GOOGLE_API_KEY"]
        except Exception:
            key = ""
    return key.strip()


@st.cache_resource
def _result_cache() -> dict:
    return {}


def analyse_file(file_name: str, data: bytes, on_stage=None) -> dict:
    """Analyse a file, remembering the last 20 results by content so re-running
    the page does not call Gemini again. A manual cache is used because
    st.cache_data forbids the progress callback (an st call) inside it."""
    cache = _result_cache()
    key = (file_name, hashlib.sha256(data).hexdigest())
    if key not in cache:
        result = core.analyse(get_api_key(), core.load_resume(file_name, data), on_stage=on_stage or (lambda _m: None))
        if len(cache) >= 20:
            cache.pop(next(iter(cache)))
        cache[key] = result
    return cache[key]


def friendly_error(error: Exception) -> str:
    message = str(error)
    lowered = message.lower()
    if any(m in lowered for m in ("503", "504", "unavailable", "high demand", "deadline")):
        return "Google's Gemini models are overloaded right now. Please try again in a minute or two."
    if "429" in message or "quota" in lowered or "resource_exhausted" in lowered:
        return "The Gemini API quota for this app is used up for the moment. Please try again in a minute."
    if "api key" in lowered or "api_key_invalid" in lowered or "permission" in lowered or "403" in message:
        return "The Gemini API key was rejected. Check the GOOGLE_API_KEY secret."
    return f"Something went wrong while analysing the resume: {message}"


def text_or_dash(value) -> str:
    return str(value) if value not in (None, "", []) else "—"


def date_range(start, end) -> str:
    parts = [str(p) for p in (start, end) if p]
    return " – ".join(parts)


def onet_caption(match: dict | None) -> None:
    if match:
        st.caption(f"Standard title (O*NET {match['code']}): **{match['title']}**  ·  match {match['score']:.2f}")


def render_header(profile: dict) -> None:
    st.header(text_or_dash(profile.get("name")))
    contact = [profile.get("email"), profile.get("phone"), profile.get("location")]
    contact = [str(c) for c in contact if c]
    if contact:
        st.write(" · ".join(contact))
    if profile["links"]:
        st.caption(" · ".join(str(link) for link in profile["links"]))
    if profile.get("summary"):
        st.info(profile["summary"])


def render_metrics(profile: dict, career: dict) -> None:
    skill_count = sum(len(items) for items in profile["skills"].values())
    columns = st.columns(4)
    columns[0].metric("Roles", len(profile["experience"]))
    columns[1].metric("Projects", len(profile["projects"]))
    columns[2].metric("Skills", skill_count)
    columns[3].metric("Experience", text_or_dash(career.get("total_experience")))


def render_experience(profile: dict) -> None:
    if not profile["experience"]:
        st.write("No work experience found in this resume.")
    for job in profile["experience"]:
        with st.container(border=True):
            st.subheader(f"{text_or_dash(job.get('title'))} · {text_or_dash(job.get('company'))}", divider=False)
            meta = [date_range(job.get("start"), job.get("end")), job.get("location")]
            meta = [str(m) for m in meta if m]
            if meta:
                st.caption(" · ".join(meta))
            for point in job["highlights"]:
                st.markdown(f"- {point}")
            onet_caption(job.get("standard_title"))


def render_education(profile: dict) -> None:
    if not profile["education"]:
        st.write("No education details found in this resume.")
        return
    rows = [
        {
            "Degree": text_or_dash(item.get("degree")),
            "Institution": text_or_dash(item.get("institution")),
            "Period": text_or_dash(date_range(item.get("start"), item.get("end"))),
            "Score": text_or_dash(item.get("score")),
        }
        for item in profile["education"]
    ]
    st.dataframe(rows, hide_index=True)


def render_projects(profile: dict) -> None:
    if not profile["projects"]:
        st.write("No projects found in this resume.")
    for project in profile["projects"]:
        with st.container(border=True):
            st.subheader(text_or_dash(project.get("name")), divider=False)
            if project["tech"]:
                st.caption(" · ".join(project["tech"]))
            if project.get("description"):
                st.write(project["description"])


def render_skills(profile: dict) -> None:
    if not profile["skills"]:
        st.write("No skills found in this resume.")
    for category, items in profile["skills"].items():
        st.markdown(f"**{category}**")
        st.markdown(" ".join(f"`{item}`" for item in items))
    for title, key in (("Certifications", "certifications"), ("Achievements", "achievements")):
        if profile[key]:
            st.markdown(f"**{title}**")
            for item in profile[key]:
                st.markdown(f"- {item}")


def render_career(career: dict) -> None:
    if career["career_path"]:
        st.markdown("**Career path** (earliest to latest)")
        steps = []
        for step in career["career_path"]:
            label = f"**{text_or_dash(step.get('title'))}** at {text_or_dash(step.get('company'))}"
            if step.get("start"):
                label += f" ({step['start']})"
            steps.append(label)
        st.markdown("  →  ".join(steps))
    else:
        st.write("No dated roles found, so a career path could not be drawn.")

    if career["strengths"]:
        st.markdown("**Strengths**")
        for item in career["strengths"]:
            st.markdown(f"- {item}")

    if career["suggested_roles"]:
        st.markdown("**Roles this resume fits**")
        for item in career["suggested_roles"]:
            st.markdown(f"- {item['role']}")
            onet_caption(item.get("standard_title"))


def render_result(result: dict, file_name: str) -> None:
    profile, career = result["profile"], result["career"]
    render_header(profile)
    render_metrics(profile, career)

    tabs = st.tabs(["Experience", "Education", "Projects", "Skills", "Career path", "Raw text"])
    with tabs[0]:
        render_experience(profile)
    with tabs[1]:
        render_education(profile)
    with tabs[2]:
        render_projects(profile)
    with tabs[3]:
        render_skills(profile)
    with tabs[4]:
        render_career(career)
    with tabs[5]:
        st.text(result["resume_text"])

    export = {"profile": profile, "career": career}
    st.download_button(
        "Download as JSON",
        data=json.dumps(export, indent=2, ensure_ascii=False),
        file_name=f"{file_name.rsplit('.', 1)[0]}_parsed.json",
        mime="application/json",
    )
    st.caption(f"Analysed with {result['model']}")


st.title("🌟 ResuMAGIC")
st.write("Upload a resume and get the education, experience, projects, skills and career path as clean, structured data.")

with st.sidebar:
    st.subheader("How it works")
    st.markdown(
        "1. The file is read into text (photos are read by Gemini).\n"
        "2. A chained Gemini pipeline extracts the details, then analyses the career.\n"
        "3. Job titles are matched to standard O*NET occupation titles."
    )
    st.subheader("Supported files")
    st.write("PDF, DOCX, HTML, TXT, JPG, PNG")
    st.subheader("Privacy")
    st.write("The resume is sent to Google Gemini for analysis. This app does not store it.")

if not get_api_key():
    st.error("Missing GOOGLE_API_KEY. Set it as an environment variable or in Streamlit secrets.")
    st.stop()

uploaded = st.file_uploader("Upload a resume", type=core.SUPPORTED_TYPES)

if uploaded is None:
    st.info("Upload a PDF, DOCX, HTML, TXT, JPG or PNG file to begin.")
else:
    status = st.status("Reading and analysing the resume...", expanded=True)

    def report(message: str) -> None:
        status.write(message)

    try:
        analysis = analyse_file(uploaded.name, uploaded.getvalue(), on_stage=report)
    except core.ResumeError as error:
        status.update(label="Could not read this file", state="error")
        st.warning(str(error))
    except Exception as error:  # noqa: BLE001 - show a readable message instead of a stack trace
        status.update(label="Analysis failed", state="error")
        st.error(friendly_error(error))
    else:
        status.update(label="Done", state="complete", expanded=False)
        render_result(analysis, uploaded.name)
