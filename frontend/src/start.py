Terminal 1 - Backend API:
cd /path/to/IKIGAI-HACKATHON/synthguard_backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
Terminal 2 - Frontend (Streamlit):
cd /path/to/IKIGAI-HACKATHON/ikigai-hackathon
pip install -r requirements.txt
streamlit run app.py
# Opens at http://localhost:8501
Terminal 3 - Frontend (React/Vite) if separate:
cd /path/to/IKIGAI-HACKATHON/frontend
npm install
npm run dev
