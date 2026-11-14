# ResuMAGIC 🌟

Upload a resume and get it back as clean, structured data: contact details, education, work experience, projects, skills, and a career path, with each job title matched to a standard O*NET occupation.

**Live app:** https://resumagic-codingrookies.streamlit.app/

## What it does

- **Reads 6 file types:** PDF, DOCX, HTML, TXT, and photos of a resume (JPG, PNG). Photos are read by Gemini itself, so no separate OCR engine is needed.
- **Extracts structured data:** a chained Gemini pipeline (built with LangChain) first pulls out the profile as JSON, then analyses the career from that JSON (career path, total experience, strengths, roles the candidate fits).
- **Standardises job titles:** every job title is matched against 1,016 O*NET occupations using TF-IDF similarity over titles and descriptions, so "Backend Engineer Intern" maps to "Software Developers".
- **Clear output:** a profile header, summary numbers, one tab per section, and a button to download everything as JSON.
- **Keeps working when models change:** the app asks the Gemini API which models exist and picks the newest stable Flash model, instead of hard-coding a name that Google may retire.

## How it works

```
file ──► text ──► [Gemini] profile JSON ──► [Gemini] career analysis ──► O*NET title match ──► UI
          ▲
  photo ──┘ (transcribed by Gemini first)
```

| File | Purpose |
|---|---|
| `streamlit_app.py` | The Streamlit UI |
| `core.py` | File reading, the Gemini chain, and O*NET title matching (no UI code) |
| `test_two.csv` | O*NET occupation titles and descriptions |

## Run it locally

Requires Python 3.10 or newer and a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

```bash
git clone https://github.com/ayaan-goel/MINeD_24
cd MINeD_24
pip install -r requirements.txt
```

Create `.streamlit/secrets.toml` with your key (this file is git-ignored):

```toml
GOOGLE_API_KEY = "your-key-here"
```

Then start the app:

```bash
streamlit run streamlit_app.py
```

It opens at http://localhost:8501.

## Configuration

| Name | Required | Meaning |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Gemini API key. Read from the environment or from Streamlit secrets. |
| `GEMINI_MODEL` | No | Force a specific model (for example `gemini-2.5-flash`). By default the newest stable Flash model is chosen automatically. |

On Streamlit Community Cloud, set these under **App settings → Secrets**.

## Privacy

The uploaded resume is sent to Google Gemini for analysis. The app itself does not store it.

## Built with

Streamlit · LangChain · Google Gemini · pdfplumber · docx2txt · BeautifulSoup · pandas

## License

MIT
