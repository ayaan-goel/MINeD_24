"""Core logic for ResuMAGIC.

Three parts, all independent of the Streamlit UI so they can be tested alone:

1. Reading a resume file (PDF, DOCX, HTML, TXT, or a photo) into text.
2. A chained Gemini pipeline (LangChain) that turns the text into structured data.
3. Matching free-text job titles to standard O*NET occupation titles.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ONET_CSV = Path(__file__).with_name("test_two.csv")

IMAGE_TYPES = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}
SUPPORTED_TYPES = ["pdf", "docx", "html", "htm", "txt", *IMAGE_TYPES]

# Enough for a long multi-page resume, and keeps a huge file from blowing the request.
MAX_RESUME_CHARS = 30_000


class ResumeError(Exception):
    """A problem the user can fix; the message is shown as-is in the UI."""


# --------------------------------------------------------------------------
# 1. Reading the file
# --------------------------------------------------------------------------

@dataclass
class ResumeFile:
    text: str = ""
    image_b64: str = ""
    image_mime: str = ""

    @property
    def is_image(self) -> bool:
        return bool(self.image_b64)


def _tidy(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def load_resume(file_name: str, data: bytes) -> ResumeFile:
    """Turn an uploaded file into text (or an image for Gemini to read)."""
    ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""

    if ext in IMAGE_TYPES:
        return ResumeFile(image_b64=base64.b64encode(data).decode(), image_mime=IMAGE_TYPES[ext])

    if ext == "pdf":
        import pdfplumber

        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                text = "\n".join((page.extract_text() or "") for page in pdf.pages)
        except Exception as error:
            raise ResumeError("This PDF could not be opened. It may be damaged or password-protected.") from error
        if not text.strip():
            raise ResumeError(
                "This PDF has no selectable text (it looks like a scan). "
                "Upload it as a JPG or PNG image instead."
            )
    elif ext == "docx":
        import docx2txt

        text = docx2txt.process(io.BytesIO(data))
    elif ext in ("html", "htm"):
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(data, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    elif ext == "txt":
        text = data.decode("utf-8", errors="ignore")
    else:
        raise ResumeError(f"Unsupported file type: .{ext or '?'}")

    text = _tidy(text)
    if len(text) < 50:
        raise ResumeError("Could not find enough text in this file to analyse.")
    return ResumeFile(text=text[:MAX_RESUME_CHARS])


# --------------------------------------------------------------------------
# 2. Gemini pipeline
# --------------------------------------------------------------------------

MODELS_URL = "https://generativelanguage.googleapis.com/v1beta/models?pageSize=200"
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-2.5-flash", "gemini-2.0-flash"]


def rank_models(models: list[dict]) -> list[str]:
    """Pick usable text models from the API's model list, newest stable Flash first."""
    stable: list[tuple[float, str]] = []
    has_latest_alias = False
    for model in models:
        if "generateContent" not in model.get("supportedGenerationMethods", []):
            continue
        name = model.get("name", "").removeprefix("models/")
        if name == "gemini-flash-latest":
            has_latest_alias = True
        match = re.fullmatch(r"gemini-(\d+(?:\.\d+)?)-flash", name)
        if match:
            stable.append((float(match.group(1)), name))
    ranked = [name for _, name in sorted(stable, reverse=True)]
    if has_latest_alias:
        ranked.append("gemini-flash-latest")
    return ranked


def candidate_models(api_key: str) -> list[str]:
    """Models to try, in order.

    Google retires model names over time (the original `gemini-pro` is gone), so
    instead of hard-coding one we ask the API what exists. `GEMINI_MODEL` overrides.
    """
    names: list[str] = []
    override = os.getenv("GEMINI_MODEL", "").strip()
    if override:
        names.append(override)
    try:
        request = urllib.request.Request(MODELS_URL, headers={"x-goog-api-key": api_key})
        with urllib.request.urlopen(request, timeout=10) as response:
            names += rank_models(json.load(response).get("models", []))
    except Exception:
        pass  # offline or listing blocked: fall through to the defaults below
    names += FALLBACK_MODELS
    return list(dict.fromkeys(names))


PROFILE_SCHEMA = """{
  "name": string or null,
  "email": string or null,
  "phone": string or null,
  "location": string or null,
  "links": [string],
  "summary": string,
  "education": [{"degree": string, "institution": string, "start": string or null, "end": string or null, "score": string or null}],
  "experience": [{"title": string, "company": string, "start": string or null, "end": string or null, "location": string or null, "highlights": [string]}],
  "projects": [{"name": string, "tech": [string], "description": string}],
  "skills": {"<category name>": [string]},
  "certifications": [string],
  "achievements": [string]
}"""

CAREER_SCHEMA = """{
  "career_path": [{"title": string, "company": string, "start": string or null}],
  "total_experience": string or null,
  "strengths": [string],
  "suggested_roles": [string]
}"""

TRANSCRIBE_INSTRUCTION = (
    "This image is a resume. Transcribe all of its text exactly as written, top to bottom, "
    "keeping section headings and line breaks. Output only the transcribed text."
)

EXTRACT_SYSTEM = (
    "You extract information from resumes. Use only what is written in the resume; never "
    "guess or invent. If something is missing, use null or an empty list. "
    "Keep dates as written (for example 'Jun 2024'). "
    "'summary' is two neutral sentences describing the candidate. "
    "'highlights' are short bullet points copied or lightly shortened from the resume. "
    "Group 'skills' under the categories the resume uses, or sensible ones such as "
    "Languages, Frameworks, Databases, Tools. "
    "Reply with a single JSON object and nothing else, in exactly this shape:\n{schema}"
)

CAREER_SYSTEM = (
    "You analyse a candidate's career from structured resume data. "
    "'career_path' lists every job or internship ordered from earliest to latest start date. "
    "'total_experience' is the combined length of the roles (for example '1 yr 4 mos'), or null "
    "if the dates are not clear enough. "
    "'strengths' are up to 4 short phrases backed by the data. "
    "'suggested_roles' are up to 4 standard job titles this candidate fits now. "
    "Reply with a single JSON object and nothing else, in exactly this shape:\n{schema}"
)


def build_chain(llm):
    """Chain: resume text -> structured profile -> career analysis."""
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    extract = (
        ChatPromptTemplate.from_messages([("system", EXTRACT_SYSTEM), ("human", "Resume:\n\n{resume}")])
        | llm
        | JsonOutputParser()
    )
    career = (
        RunnableLambda(
            lambda state: {
                "schema": CAREER_SCHEMA,
                "data": json.dumps(
                    {key: state["profile"].get(key) for key in ("experience", "projects", "skills", "education")}
                ),
            }
        )
        | ChatPromptTemplate.from_messages([("system", CAREER_SYSTEM), ("human", "Resume data:\n\n{data}")])
        | llm
        | JsonOutputParser()
    )
    return RunnablePassthrough.assign(profile=extract) | RunnablePassthrough.assign(career=career)


def transcribe_image(llm, resume: ResumeFile) -> str:
    from langchain_core.messages import HumanMessage
    from langchain_core.output_parsers import StrOutputParser

    message = HumanMessage(
        content=[
            {"type": "text", "text": TRANSCRIBE_INSTRUCTION},
            {"type": "image_url", "image_url": f"data:{resume.image_mime};base64,{resume.image_b64}"},
        ]
    )
    text = _tidy((llm | StrOutputParser()).invoke([message]))
    if len(text) < 50:
        raise ResumeError("Could not read any resume text from this image.")
    return text[:MAX_RESUME_CHARS]


def _as_list(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _normalise(profile: dict, career: dict) -> dict:
    """Make sure every expected key exists with the right type, whatever the model returned."""
    profile = profile if isinstance(profile, dict) else {}
    career = career if isinstance(career, dict) else {}
    for key in ("links", "education", "experience", "projects", "certifications", "achievements"):
        profile[key] = _as_list(profile.get(key))
    for key in ("education", "experience", "projects"):
        profile[key] = [item for item in profile[key] if isinstance(item, dict)]
    skills = profile.get("skills")
    if isinstance(skills, list):
        skills = {"Skills": skills}
    profile["skills"] = {
        str(category): [str(s) for s in _as_list(items)]
        for category, items in (skills or {}).items()
        if _as_list(items)
    }
    for job in profile["experience"]:
        job["highlights"] = [str(h) for h in _as_list(job.get("highlights"))]
    for project in profile["projects"]:
        project["tech"] = [str(t) for t in _as_list(project.get("tech"))]
    for key in ("career_path", "strengths", "suggested_roles"):
        career[key] = _as_list(career.get(key))
    career["career_path"] = [step for step in career["career_path"] if isinstance(step, dict)]
    return {"profile": profile, "career": career}


def run_pipeline(llm, resume: ResumeFile) -> dict:
    """Run the full chain with one model and add O*NET matches."""
    text = transcribe_image(llm, resume) if resume.is_image else resume.text
    state = build_chain(llm).invoke({"resume": text, "schema": PROFILE_SCHEMA})
    result = _normalise(state.get("profile"), state.get("career"))

    matcher = get_matcher()
    for job in result["profile"]["experience"]:
        job["standard_title"] = matcher.match(str(job.get("title") or ""), " ".join(job["highlights"]))
    result["career"]["suggested_roles"] = [
        {"role": str(role), "standard_title": matcher.match(str(role))}
        for role in result["career"]["suggested_roles"]
    ]
    result["resume_text"] = text
    return result


def _model_unavailable(error: Exception) -> bool:
    message = str(error).lower()
    return "404" in message or "not found" in message or "not supported" in message


def analyse(api_key: str, resume: ResumeFile, make_llm=None) -> dict:
    """Analyse a resume, moving to the next model if one has been retired."""
    if make_llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        def make_llm(model: str):
            return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.1)

    last_error: Exception | None = None
    for model in candidate_models(api_key):
        try:
            result = run_pipeline(make_llm(model), resume)
            result["model"] = model
            return result
        except ResumeError:
            raise
        except Exception as error:  # noqa: BLE001 - decide below whether to try another model
            if not _model_unavailable(error):
                raise
            last_error = error
    raise RuntimeError(f"No usable Gemini model was found for this API key. Last error: {last_error}")


# --------------------------------------------------------------------------
# 3. O*NET title matching
# --------------------------------------------------------------------------

_STOPWORDS = set(
    "a an and are as at be by for from in into is it its of on or other such that the their this "
    "to using with within may including e g etc all any work works perform performs related".split()
)
# Words that describe seniority or contract type, not the occupation itself.
_LEVEL_WORDS = set(
    "intern internship trainee junior jr senior sr lead principal staff associate assistant "
    "freelance contract part full time graduate apprentice i ii iii".split()
)
_SYNONYMS = {"engineer": "developer", "programmer": "developer", "swe": "software", "sde": "software"}


def _stem(word: str) -> str:
    for suffix in ("ies", "ing", "ers", "er", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    # "nurse" and "nurses" should end up the same
    return word[:-1] if word.endswith("e") and len(word) > 4 else word


def _tokens(text: str) -> list[str]:
    text = re.sub(r"\bhr\b", "human resources", text.lower())
    words = re.findall(r"[a-z][a-z+#]*", text)
    return [
        _stem(_SYNONYMS.get(word, word))
        for word in words
        if word not in _STOPWORDS and word not in _LEVEL_WORDS and len(word) > 1
    ]


class TitleMatcher:
    """Match a free-text job title to the closest O*NET occupation.

    Two TF-IDF cosine scores are combined: the job title against the O*NET
    titles, and the title plus job highlights against the O*NET descriptions.
    The second one lets "Backend Engineer Intern" still land on
    "Software Developers" even though the titles share few words.
    """

    TITLE_WEIGHT = 0.65
    MIN_SCORE = 0.3

    def __init__(self, table: pd.DataFrame):
        self.titles = table["standerlised_tittle"].astype(str).tolist()
        self.codes = table["o_net_code"].astype(str).tolist()
        title_docs = [Counter(_tokens(title)) for title in self.titles]
        full_docs = [
            Counter(_tokens(title) + _tokens(description))
            for title, description in zip(self.titles, table["job_description"].astype(str))
        ]
        self.title_idf = self._idf(title_docs)
        self.full_idf = self._idf(full_docs)
        self.title_vectors = [self._vector(doc, self.title_idf) for doc in title_docs]
        self.full_vectors = [self._vector(doc, self.full_idf) for doc in full_docs]

    @staticmethod
    def _idf(documents: list[Counter]) -> dict[str, float]:
        frequency = Counter(term for doc in documents for term in doc)
        total = len(documents)
        return {term: math.log((1 + total) / (1 + count)) + 1 for term, count in frequency.items()}

    @staticmethod
    def _vector(counts: Counter, idf: dict[str, float]) -> dict[str, float]:
        weights = {term: (1 + math.log(n)) * idf[term] for term, n in counts.items() if term in idf}
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1.0
        return {term: w / norm for term, w in weights.items()}

    @staticmethod
    def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
        small, large = (a, b) if len(a) < len(b) else (b, a)
        return sum(weight * large.get(term, 0.0) for term, weight in small.items())

    def match(self, title: str, context: str = "") -> dict | None:
        title_tokens = _tokens(title)
        if not title_tokens:
            return None
        title_query = self._vector(Counter(title_tokens), self.title_idf)
        full_query = self._vector(Counter(title_tokens * 2 + _tokens(context)), self.full_idf)
        best_score, best_index = 0.0, -1
        for index in range(len(self.titles)):
            score = self.TITLE_WEIGHT * self._cosine(title_query, self.title_vectors[index]) + (
                1 - self.TITLE_WEIGHT
            ) * self._cosine(full_query, self.full_vectors[index])
            if score > best_score:
                best_score, best_index = score, index
        if best_index < 0 or best_score < self.MIN_SCORE:
            return None
        return {
            "title": self.titles[best_index],
            "code": self.codes[best_index],
            "score": round(best_score, 2),
        }

_matcher: TitleMatcher | None = None


def get_matcher() -> TitleMatcher:
    global _matcher
    if _matcher is None:
        _matcher = TitleMatcher(pd.read_csv(ONET_CSV))
    return _matcher
