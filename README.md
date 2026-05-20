# Anent — Six AI Streamlit Demos

A portfolio of six self-contained Streamlit demos showcasing different
applied-AI patterns. Each demo lives in its own folder with its own
`app.py` and `requirements.txt`, so they can be deployed independently
to Streamlit Community Cloud (or any other Streamlit host).

## The six demos

| # | Folder | What it shows | Suggested URL slug |
|---|---|---|---|
| 1 | [`AIconsultant/`](AIconsultant/) | Lead Consultant agent running a strict **7-step business AI consulting process** (assess landscape → opportunities → research → driver tree → prioritisation → pyramid recommendations → FAST plan), with chat-based refinement. | `anent-ai-consultant` |
| 2 | [`ClaudeExcelPlugIn/`](ClaudeExcelPlugIn/) | Interactive **setup guide for the Claude for Excel plug-in** — sidebar mode switch between text walkthrough (with progress checklists) and curated video guides. | `anent-claude-excel-guide` |
| 3 | [`DocReviewAndGeneration/`](DocReviewAndGeneration/) | **Document agent** that takes natural-language instructions and calls tools (generate/edit, create/fill forms, RAG, review) against a live editable document. | `anent-doc-agent` |
| 4 | [`InvMan/`](InvMan/) | Multi-agent **real-time inventory manager** — Lead Agent + Assistant Agent with weather/social/demand/supply tools, sklearn forecasting, reasoning console. | `anent-inventory-ai` |
| 5 | [`MarketingCampaign/`](MarketingCampaign/) | Lead Consultant agent running a **4-step marketing campaign generation script** producing a full downloadable campaign document. | `anent-marketing-campaign` |
| 6 | [`VibeCodeHardening/`](VibeCodeHardening/) | **6-step wizard** that analyses vibe-coded Python, lets the user approve fixes, applies them, runs the hardened code in a sandbox, and packages a download. | `anent-code-hardening` |

## Local run (any demo)

```bash
cd <folder>
pip install -r requirements.txt
streamlit run app.py
```

## Deploying to Streamlit Community Cloud

The repo is structured so each demo deploys as its own app — Streamlit
Cloud picks the entry point per deployment.

### One-time setup

1. Create a free GitHub account if you don't have one.
2. Push this repo to GitHub (see commands below).
3. Sign in at <https://share.streamlit.io> using that GitHub account.
4. Authorize Streamlit to read your repos.

### Push to GitHub

```bash
cd /path/to/Anent
git init
git add .
git commit -m "Initial commit — six Streamlit AI demos"
# Create an empty repo at github.com/<you>/anent-demos (no README, no .gitignore).
git branch -M main
git remote add origin git@github.com:<your-username>/anent-demos.git
git push -u origin main
```

### Deploy each app

For each of the six demos, click **"New app"** on share.streamlit.io and fill in:

| Field | Value |
|---|---|
| Repository | `<your-username>/anent-demos` |
| Branch | `main` |
| Main file path | e.g. `AIconsultant/app.py` |
| App URL | the slug from the table above |

Click **Deploy**. First deploy takes ~2–3 minutes (Streamlit installs deps from
`requirements.txt`). Subsequent deploys are faster.

### Optional: set OpenAI API key as a secret

Apps 1, 3, 4, 5, 6 will use OpenAI if a key is present, otherwise fall back
to deterministic templates (the demos still work cleanly without a key).

To enable LLM features, in each app's **Settings → Secrets** on Streamlit Cloud paste:

```toml
OPENAI_API_KEY = "sk-..."
```

## Free-tier notes

- Apps sleep after ~7 days of inactivity. First visit after sleep is ~15 s cold-start.
- 1 GB RAM / 1 GB storage per app — plenty for these demos.
- Public URLs follow the pattern `https://<slug>.streamlit.app`.

## Final link block for your website

Once deployed, the URLs to embed on your site will be:

```text
https://anent-ai-consultant.streamlit.app
https://anent-claude-excel-guide.streamlit.app
https://anent-doc-agent.streamlit.app
https://anent-inventory-ai.streamlit.app
https://anent-marketing-campaign.streamlit.app
https://anent-code-hardening.streamlit.app
```

(Replace the slugs with whatever you actually chose during deploy.)
