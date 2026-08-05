# Employee Attrition Risk Predictor

## 📌 Overview
This project predicts the likelihood of employee attrition using HR data.  
It combines a **FastAPI backend** for serving predictions with a **Streamlit frontend** for an interactive user interface.

## 🚀 Features
- Interactive sliders and dropdowns for employee attributes
- Real‑time prediction of attrition risk
- Modular design: FastAPI backend + Streamlit frontend
- Debug mode for testing with sample inputs

## 🛠️ Tech Stack
- Python 3.10+
- [Streamlit](https://streamlit.io) for the frontend
- [FastAPI](https://fastapi.tiangolo.com/) for the backend
- [Uvicorn](https://www.uvicorn.org/) as the ASGI server
- [Starlette](https://www.starlette.io/) for ASGI utilities
- Machine learning model built with [Scikit-learn](https://scikit-learn.org/) or similar

## ⚙️ Installation
Clone the repository and set up a virtual environment:

```powershell
git clone <your-repo-url>
cd coding-practice
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
