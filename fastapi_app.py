"""
COSMOS-ITS Grade Prediction API (FastAPI Version)
=================================================
High-performance, asynchronous API for the COSMOS-ITS Prediction Model.
Built with FastAPI for automatic validation and documentation.

Usage:
    uvicorn fastapi_app:app --reload

Author: COSMOS-ITS Development Team
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import traceback
from cosmos_predictor_api import predict_grade

# -------------------------
# Pydantic Models (Data Validation)
# -------------------------

class CurrentCourseInput(BaseModel):
    course: str = Field(..., example="CSE2118")
    ct: float = Field(..., ge=0, le=20, description="Class Test marks (max 20)")
    assignment: float = Field(..., ge=0, le=20, description="Assignment marks (max 10-15)")
    attendance: float = Field(..., ge=0, le=10, description="Attendance marks (max 5-10)")
    mid: float = Field(..., ge=0, le=30, description="Midterm marks (max 25-30)")
    project: Optional[float] = Field(0, ge=0, le=30, description="Project marks (if applicable)")

class PreviousCourse(BaseModel):
    course: str
    ct: Optional[float] = 0
    assignment: Optional[float] = 0
    attendance: Optional[float] = 0
    mid: Optional[float] = 0
    project: Optional[float] = 0
    final: float
    grade: str
    grade_point: float

class TrimesterData(BaseModel):
    trimester: int
    gpa: float
    courses: List[PreviousCourse]

class PredictionRequest(BaseModel):
    current_course: CurrentCourseInput
    previous_trimesters: Optional[List[TrimesterData]] = None
    cgpa: Optional[float] = Field(None, ge=0.0, le=4.0)
    current_trimester: Optional[int] = Field(1, ge=1)
    use_meta_learning: Optional[bool] = True

# -------------------------
# FastAPI App Setup
# -------------------------

app = FastAPI(
    title="COSMOS-ITS Grade Prediction API",
    description="AI-powered grade prediction engine for students.",
    version="2.0.0"
)

# Enable CORS for React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (update with your React app URL in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Endpoints
# -------------------------

@app.get("/")
def root():
    """Health check and API info"""
    return {
        "service": "COSMOS-ITS Prediction API",
        "status": "active",
        "version": "2.0.0",
        "docs_url": "/docs"
    }

@app.post("/predict")
def predict_student_grade(request: PredictionRequest):
    """
    Predict the final grade for a student based on current performance and history.
    """
    try:
        # Convert Pydantic models to dicts for the legacy logic
        current_course_dict = request.current_course.dict()
        
        previous_trimesters_list = None
        if request.previous_trimesters:
            previous_trimesters_list = [t.dict() for t in request.previous_trimesters]
        
        # Call the prediction logic
        result = predict_grade(
            current_course=current_course_dict,
            previous_trimesters=previous_trimesters_list,
            cgpa=request.cgpa,
            current_trimester=request.current_trimester,
            use_meta_learning=request.use_meta_learning
        )
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Starting COSMOS-ITS FastAPI Server...")
    uvicorn.run("fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
