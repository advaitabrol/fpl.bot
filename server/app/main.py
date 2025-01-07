from fastapi import FastAPI
from app.routes import teams
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Include routes from different modules
app.include_router(teams.router, prefix="/teams", tags=["Teams"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the API"}



'''
Start Server CMD: uvicorn app.main:app --reload




Example Front End Snippet


import axios from 'axios';

const fetchTeam = async (teamName) => {
  const response = await axios.get(`http://127.0.0.1:8000/teams/${teamName}`);
  console.log(response.data);
};

fetchTeam("Manchester United");
'''