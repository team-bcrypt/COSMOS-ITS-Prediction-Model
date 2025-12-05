"""
COSMOS-ITS Grade Prediction API
================================
Handles both fresh students (0-2 trimesters) and experienced students (3+ trimesters)
with personalized meta-learning adjustments.
SUPPORTS BOTH THEORY AND LAB COURSE MARKING SCHEMES.

Author: COSMOS-ITS Development Team
Date: December 2025
"""

import json
import numpy as np
import pandas as pd
import joblib
import os
from typing import Dict, List, Optional, Tuple

# -------------------------
# CONFIGURATION
# -------------------------

MODELS_DIR = "./models"
REQUIRED_MODELS = [
    "rf_enhanced_predictor.joblib",
    "gb_predictor.joblib",
    "mlp_enhanced_predictor.joblib",
    "feature_scaler.joblib"
]

# Course clustering (same as training)
COURSE_CLUSTERS = {
    "PROG_CHAIN": ["CSE1110", "CSE1111", "CSE1115", "CSE2118", "CSE2215", "CSE2217"],
    "ENGLISH": ["ENG1011", "ENG1013"],
    "MATH": ["MATH1151", "MATH2183", "MATH2201", "MATH2205"],
    "THEORY_SE": ["CSE3411", "CSE3421", "PMG4101"],
    "LAB_CORE": ["CSE1112", "CSE1116", "CSE2118", "CSE2216", "CSE2218"],
    "LAB_THEORY_PAIR": {
        "CSE3521": "CSE3522", "CSE3411": "CSE3412", "CSE3421": "CSE3422",
        "PHY2105": "PHY2106", "CSE3711": "CSE3712", "CSE3811": "CSE3812",
    },
    "GEN_ED": ["BDS1201", "SOC2101", "BIO3105", "GEDOPT1", "GEDOPT2", "GEDOPT3"],
    "PHY": ["PHY2105", "PHY2106"],
    "EEE": ["EEE2113", "EEE2123", "EEE2124"],
}

# ✅ NEW: Define marking schemes for different course types
MARKING_SCHEMES = {
    # Theory courses: CT(20) + Assignment(10) + Attendance(5) + Mid(30) = 65 (usually counted as 60 or 70 pre-final)
    # We will use the sum of maxes as the denominator.
    "THEORY": {
        "ct_max": 20,
        "assignment_max": 10,
        "attendance_max": 5,
        "mid_max": 30,
        "project_max": 0,
        "pre_final_total": 60  # Standard denominator for Theory
    },
    # Lab courses: Variable marking
    # Example: Attendance(10) + Assignment(15) + Mid(25) + CT(20) + Project(10) = 80
    "LAB": {
        "ct_max": 20,
        "assignment_max": 15,
        "attendance_max": 10,
        "mid_max": 25,
        "project_max": 20,
        "pre_final_total": 80  # Standard denominator for Lab (can be dynamic)
    }
}

# Feature columns (must match training exactly)
FEATURE_COLS = [
    "cgpa", "cgpa_scaled", "current_trimester", "prev_trimester_gpa", "gpa_trend", "cgpa_gpa_diff",
    "prog_chain_avg", "english_avg", "math_avg", "lab_core_avg", "theory_se_avg", "gen_ed_avg",
    "overall_prev_avg", "prog_chain_std",
    "curr_ct", "curr_assignment", "curr_attendance", "curr_mid", "curr_total", "curr_percentage"
]

# Grade conversion
GRADE_MAPPING = [
    (90, 999, "A", 4.00),  # Allow scores > 100 to be A
    (86, 89, "A-", 3.67), (82, 85, "B+", 3.33), (78, 81, "B", 3.00),
    (74, 77, "B-", 2.67), (70, 73, "C+", 2.33), (66, 69, "C", 2.00), (62, 65, "C-", 1.67),
    (58, 61, "D+", 1.33), (55, 57, "D", 1.00), (0, 54, "F", 0.00)
]

# -------------------------
# GRADE CONVERSION
# -------------------------

def score_to_grade(score: float) -> Tuple[str, float]:
    """Convert numerical score to letter grade and grade point"""
    # Safety clip
    if score > 100:
        score = 100
    elif score < 0:
        score = 0
        
    # Round to nearest integer to handle floating point gaps in GRADE_MAPPING
    # e.g., 77.44 -> 77 (B-), 77.6 -> 78 (B)
    score_rounded = int(score + 0.5)
        
    for lo, hi, grade, gp in GRADE_MAPPING:
        if lo <= score_rounded <= hi:
            return grade, gp
    return "F", 0.0


def calculate_max_possible_score(current_course: Dict) -> float:
    """
    Calculate the maximum possible score a student can achieve
    based on marks already lost in continuous assessment.
    Assumes 40/40 in Final Exam.
    """
    course_code = current_course.get("course", "")
    curr_mid = current_course.get("mid", 0)
    curr_project = current_course.get("project", 0)
    
    # Determine Scheme (Logic matches build_feature_vector)
    is_lab_by_config = "LAB" in course_code or course_code in COURSE_CLUSTERS["LAB_CORE"]
    
    if curr_project > 0:
        is_lab = True
    elif curr_mid > 25:
        is_lab = False
    else:
        is_lab = is_lab_by_config
        
    scheme = MARKING_SCHEMES["LAB"] if is_lab else MARKING_SCHEMES["THEORY"]
    
    lost_marks = 0.0
    
    # Only deduct marks for components that are present AND have been attempted
    # If a component score is 0, check if it's truly "not attempted" or just scored 0
    
    if "ct" in current_course:
        lost_marks += max(0, scheme["ct_max"] - current_course["ct"])
        
    if "assignment" in current_course:
        lost_marks += max(0, scheme["assignment_max"] - current_course["assignment"])
        
    if "attendance" in current_course:
        lost_marks += max(0, scheme["attendance_max"] - current_course["attendance"])
        
    if "mid" in current_course:
        lost_marks += max(0, scheme["mid_max"] - current_course["mid"])
        
    # ✅ FIX: Only count project loss if project marks were actually provided (not just 0)
    # If project is explicitly in the input AND > 0, then count the loss
    # If project is 0 or not provided, assume it hasn't happened yet
    if "project" in current_course and current_course.get("project", 0) > 0:
        lost_marks += max(0, scheme["project_max"] - current_course["project"])
    # ✅ NEW: If project is 0 but the course is a Lab, assume project hasn't been evaluated yet
    # Don't penalize the student for a future assessment
    
    return max(0.0, 100.0 - lost_marks)


# -------------------------
# MODEL LOADER
# -------------------------

class ModelLoader:
    """Lazy-load models only when needed"""
    
    def __init__(self, models_dir: str = MODELS_DIR):
        self.models_dir = models_dir
        self._rf = None
        self._gb = None
        self._mlp = None
        self._scaler = None
        self._check_models_exist()
    
    def _check_models_exist(self):
        """Verify all required model files exist"""
        missing = [f for f in REQUIRED_MODELS if not os.path.exists(os.path.join(self.models_dir, f))]
        if missing:
            raise FileNotFoundError(
                f"Missing model files: {missing}\n"
                f"Please train models first using COSMOS_Prediction_Model.ipynb (Cells 1-10)"
            )
    
    @property
    def rf(self):
        if self._rf is None:
            self._rf = joblib.load(os.path.join(self.models_dir, "rf_enhanced_predictor.joblib"))
        return self._rf
    
    @property
    def gb(self):
        if self._gb is None:
            self._gb = joblib.load(os.path.join(self.models_dir, "gb_predictor.joblib"))
        return self._gb
    
    @property
    def mlp(self):
        if self._mlp is None:
            self._mlp = joblib.load(os.path.join(self.models_dir, "mlp_enhanced_predictor.joblib"))
        return self._mlp
    
    @property
    def scaler(self):
        if self._scaler is None:
            self._scaler = joblib.load(os.path.join(self.models_dir, "feature_scaler.joblib"))
        return self._scaler


# -------------------------
# FEATURE EXTRACTION HELPERS
# -------------------------

def extract_cluster_avg(previous_trimesters: List[Dict], cluster_courses: List[str]) -> float:
    """Extract average score for a cluster of courses from previous trimesters"""
    scores = []
    for trimester in previous_trimesters:
        for course in trimester.get("courses", []):
            if course["course"] in cluster_courses and course.get("final") is not None:
                scores.append(course["final"])
    return np.mean(scores) if scores else np.nan


def extract_overall_avg(previous_trimesters: List[Dict]) -> float:
    """Extract overall average from all previous courses"""
    scores = []
    for trimester in previous_trimesters:
        for course in trimester.get("courses", []):
            if course.get("final") is not None:
                scores.append(course["final"])
    return np.mean(scores) if scores else np.nan


def extract_cluster_std(previous_trimesters: List[Dict], cluster_courses: List[str]) -> float:
    """Extract standard deviation for consistency measurement"""
    scores = []
    for trimester in previous_trimesters:
        for course in trimester.get("courses", []):
            if course["course"] in cluster_courses and course.get("final") is not None:
                scores.append(course["final"])
    return np.std(scores) if len(scores) > 1 else 0.0


def extract_last_gpa(previous_trimesters: List[Dict]) -> float:
    """Get the most recent trimester GPA"""
    if previous_trimesters:
        return previous_trimesters[-1].get("gpa", np.nan)
    return np.nan


def extract_gpa_trend(previous_trimesters: List[Dict]) -> float:
    """Calculate GPA trend (last - second_last)"""
    if len(previous_trimesters) >= 2:
        return previous_trimesters[-1]["gpa"] - previous_trimesters[-2]["gpa"]
    return 0.0


# -------------------------
# LAYER 1: BASE MODEL PREDICTION
# -------------------------

def build_feature_vector(
    current_course: Dict,
    previous_trimesters: Optional[List[Dict]] = None,
    cgpa: Optional[float] = None,
    current_trimester: int = 1
) -> pd.DataFrame:
    """
    Build the 20-feature vector required by the model
    
    Args:
        current_course: Dict with keys: course, ct, assignment, attendance, mid, (project optional)
        previous_trimesters: List of previous trimester data (None for fresh students)
        cgpa: Current CGPA (None will estimate from current performance)
        current_trimester: Current trimester number
    
    Returns:
        DataFrame with 20 features ready for prediction
    """
    
    # Extract current semester components
    curr_ct = current_course.get("ct", 0)
    curr_assignment = current_course.get("assignment", 0)
    curr_attendance = current_course.get("attendance", 0)
    curr_mid = current_course.get("mid", 0)
    curr_project = current_course.get("project", 0)  # For courses with projects
    
    # Determine course type and marking scheme
    course_code = current_course.get("course", "")
    
    # Heuristic Refinement:
    # 1. If project marks exist (>0), treat as Lab/Project.
    # 2. If Mid marks > 25, treat as Theory (Labs usually max at 25).
    # 3. Otherwise, rely on Course Code / Clusters.
    
    is_lab_by_config = "LAB" in course_code or course_code in COURSE_CLUSTERS["LAB_CORE"]
    
    if curr_project > 0:
        is_lab = True
    elif curr_mid > 25:
        is_lab = False
    else:
        is_lab = is_lab_by_config
    
    # Calculate total earned so far
    curr_total = curr_ct + curr_assignment + curr_attendance + curr_mid + curr_project
    
    # Calculate Percentage dynamically based on course type
    if is_lab:
        # For Labs: Denominator is usually higher (e.g., 80)
        # The user specified variable components for labs (Att=10, Ass=15, Mid=25, CT=20, Proj=20 etc.)
        # Since we don't get the 'max' marks from frontend, we assume a standard Lab is out of 80 pre-final.
        # If the student's total is higher, we adjust.
        denominator = 80.0
    else:
        # For Theory: Standard is 60 (CT=20, Ass=10, Att=5, Mid=25/30)
        denominator = 60.0
        
    # Safety: If student somehow got more than denominator, cap percentage or adjust denominator
    if curr_total > denominator:
        denominator = curr_total  # Assume it's out of what they got at minimum
        
    curr_percentage = (curr_total / denominator) * 100
    
    # Initialize features
    features = {}
    
    # -------------------------
    # Handle FRESH STUDENTS (0-2 trimesters)
    # -------------------------
    if previous_trimesters is None or len(previous_trimesters) == 0:
        # Estimate CGPA from current performance
        if cgpa is None:
            # Estimate: if student earned 75% of current marks, assume 3.0 CGPA
            cgpa = 1.0 + (curr_percentage / 100.0) * 3.0
            cgpa = np.clip(cgpa, 1.0, 4.0)
        
        # Use current performance as proxy for historical data
        estimated_score = (curr_percentage / 100.0) * 100  # Convert to 0-100 scale
        
        features = {
            "cgpa": cgpa,
            "cgpa_scaled": (cgpa / 4.0) * 100,
            "current_trimester": current_trimester,
            "prev_trimester_gpa": cgpa,  # Use CGPA as estimate
            "gpa_trend": 0.0,  # No trend for fresh students
            "cgpa_gpa_diff": 0.0,
            
            # Use current performance as estimate for all clusters
            "prog_chain_avg": estimated_score,
            "english_avg": estimated_score * 0.95,  # Slight adjustment
            "math_avg": estimated_score * 0.92,
            "lab_core_avg": estimated_score * 1.03,  # Labs tend higher
            "theory_se_avg": estimated_score * 0.97,
            "gen_ed_avg": estimated_score * 0.98,
            "overall_prev_avg": estimated_score,
            "prog_chain_std": 0.0,  # No variation for fresh students
            
            "curr_ct": curr_ct,
            "curr_assignment": curr_assignment,
            "curr_attendance": curr_attendance,
            "curr_mid": curr_mid,
            "curr_total": curr_total,
            "curr_percentage": curr_percentage
        }
    
    # -------------------------
    # Handle EXPERIENCED STUDENTS (3+ trimesters)
    # -------------------------
    else:
        # Calculate CGPA if not provided
        if cgpa is None:
            all_gpas = [t["gpa"] for t in previous_trimesters]
            cgpa = np.mean(all_gpas)
        
        # Extract historical features
        prog_chain_avg = extract_cluster_avg(previous_trimesters, COURSE_CLUSTERS["PROG_CHAIN"])
        english_avg = extract_cluster_avg(previous_trimesters, COURSE_CLUSTERS["ENGLISH"])
        math_avg = extract_cluster_avg(previous_trimesters, COURSE_CLUSTERS["MATH"])
        lab_core_avg = extract_cluster_avg(previous_trimesters, COURSE_CLUSTERS["LAB_CORE"])
        theory_se_avg = extract_cluster_avg(previous_trimesters, COURSE_CLUSTERS["THEORY_SE"])
        gen_ed_avg = extract_cluster_avg(previous_trimesters, COURSE_CLUSTERS["GEN_ED"])
        overall_prev_avg = extract_overall_avg(previous_trimesters)
        prog_chain_std = extract_cluster_std(previous_trimesters, COURSE_CLUSTERS["PROG_CHAIN"])
        
        prev_trimester_gpa = extract_last_gpa(previous_trimesters)
        gpa_trend = extract_gpa_trend(previous_trimesters)
        
        # Fill missing values with overall average
        overall_avg_value = overall_prev_avg if not np.isnan(overall_prev_avg) else (cgpa / 4.0) * 100
        
        features = {
            "cgpa": cgpa,
            "cgpa_scaled": (cgpa / 4.0) * 100,
            "current_trimester": current_trimester,
            "prev_trimester_gpa": prev_trimester_gpa if not np.isnan(prev_trimester_gpa) else cgpa,
            "gpa_trend": gpa_trend,
            "cgpa_gpa_diff": cgpa - (prev_trimester_gpa if not np.isnan(prev_trimester_gpa) else cgpa),
            
            "prog_chain_avg": prog_chain_avg if not np.isnan(prog_chain_avg) else overall_avg_value,
            "english_avg": english_avg if not np.isnan(english_avg) else overall_avg_value,
            "math_avg": math_avg if not np.isnan(math_avg) else overall_avg_value,
            "lab_core_avg": lab_core_avg if not np.isnan(lab_core_avg) else overall_avg_value,
            "theory_se_avg": theory_se_avg if not np.isnan(theory_se_avg) else overall_avg_value,
            "gen_ed_avg": gen_ed_avg if not np.isnan(gen_ed_avg) else overall_avg_value,
            "overall_prev_avg": overall_avg_value,
            "prog_chain_std": prog_chain_std,
            
            "curr_ct": curr_ct,
            "curr_assignment": curr_assignment,
            "curr_attendance": curr_attendance,
            "curr_mid": curr_mid,
            "curr_total": curr_total,
            "curr_percentage": curr_percentage
        }
    
    # Create DataFrame with correct column order
    return pd.DataFrame([features])[FEATURE_COLS]


def predict_with_hybrid(feature_vector: pd.DataFrame, models: ModelLoader) -> Dict:
    """
    Layer 1: Base prediction using hybrid GB + MLP strategy
    
    Returns:
        Dict with prediction, grade, confidence, and model used
    """
    
    # Get predictions from both models
    raw_gb_pred = models.gb.predict(feature_vector)[0]
    
    # Scale features for MLP
    X_scaled = models.scaler.transform(feature_vector)
    raw_mlp_pred = models.mlp.predict(X_scaled)[0]
    
    # ✅ FIX: Clip predictions to valid range [0, 100] immediately
    # This prevents "F" grades for scores like 102.5 or 121.5
    gb_pred = np.clip(raw_gb_pred, 0, 100)
    mlp_pred = np.clip(raw_mlp_pred, 0, 100)
    
    # Hybrid strategy: Use MLP for edge grades (A, A-, D, F), GB for others
    gb_grade, _ = score_to_grade(gb_pred)
    mlp_grade, _ = score_to_grade(mlp_pred)
    
    # Edge grades: A, A-, D, F
    edge_grades = ["A", "A-", "D", "F"]
    
    # Confidence Logic
    diff = abs(gb_pred - mlp_pred)
    
    if diff < 5.0:
        confidence = "High"
    elif diff < 10.0:
        confidence = "Medium"
    else:
        confidence = "Low"

    # -------------------------
    # MODEL SELECTION LOGIC
    # -------------------------
    
    # 1. Low Confidence / High Disagreement Handling
    # If models disagree significantly (>10 marks), we trust the one closer to current performance
    # This prevents MLP from "hallucinating" 100% when a student has 70%
    if confidence == "Low":
        current_perf = feature_vector["curr_percentage"].values[0]
        
        # High Performance Override: If marks are very close to max (>= 95%), pick the higher prediction
        # This covers cases like 57-60/60 (Theory) or high Lab scores.
        if current_perf >= 95.0:
            if gb_pred > mlp_pred:
                final_score = gb_pred
                model_used = "Gradient Boosting (High Perf Override)"
            else:
                final_score = mlp_pred
                model_used = "MLP Neural Network (High Perf Override)"
        else:
            gb_diff = abs(gb_pred - current_perf)
            mlp_diff = abs(mlp_pred - current_perf)
            
            if gb_diff < mlp_diff:
                final_score = gb_pred
                model_used = "Gradient Boosting (Sanity Check)"
            else:
                final_score = mlp_pred
                model_used = "MLP Neural Network (Sanity Check)"
            
    # 2. Standard Hybrid Strategy (High/Medium Confidence)
    else:
        if mlp_grade in edge_grades:
            final_score = mlp_pred
            model_used = "MLP Neural Network"
        else:
            final_score = gb_pred
            model_used = "Gradient Boosting"
    
    # -------------------------
    # 3. MOMENTUM CORRECTION (High Current Performance)
    # -------------------------
    # If student is performing exceptionally well in current trimester (>85%),
    # but models are predicting significantly lower (likely due to history),
    # we boost the score towards current performance.
    
    current_perf = feature_vector["curr_percentage"].values[0]
    
    if current_perf > 85.0 and final_score < (current_perf - 5.0):
        # Calculate momentum boost
        # We trust current performance more when it's high
        momentum_factor = 0.8  
        boost = (current_perf - final_score) * momentum_factor
        
        final_score += boost
        final_score = np.clip(final_score, 0, 100)
        
        # Update metadata to reflect this boost
        model_used += " + Momentum Boost"

    final_grade, grade_point = score_to_grade(final_score)
    
    return {
        "predicted_score": round(float(final_score), 2),
        "predicted_grade": final_grade,
        "grade_point": grade_point,
        "model_used": model_used,
        "gb_prediction": round(float(gb_pred), 2),
        "mlp_prediction": round(float(mlp_pred), 2),
        "confidence": confidence,
        "raw_gb_prediction": round(float(raw_gb_pred), 2), # Debug info
        "raw_mlp_prediction": round(float(raw_mlp_pred), 2) # Debug info
    }


# -------------------------
# LAYER 2: META-LEARNING PERSONALIZATION
# -------------------------

class PersonalizedAdjuster:
    """
    Layer 2: Applies personalized adjustments based on student's behavioral patterns
    
    Analyzes:
    - Personal Performance Metrics (consistency, improvement trends)
    - Course-Type Strengths (theory vs. lab, programming vs. math)
    - Behavioral Patterns (study consistency, performance under pressure)
    """
    
    def __init__(self, previous_trimesters: List[Dict], current_course: Dict):
        self.previous_trimesters = previous_trimesters
        self.current_course = current_course
        self.course_code = current_course.get("course", "")
        
    def calculate_personal_metrics(self) -> Dict:
        """Extract personal performance patterns"""
        
        if not self.previous_trimesters or len(self.previous_trimesters) == 0:
            return {
                "consistency_score": 0.5,  # Neutral for fresh students
                "improvement_rate": 0.0,
                "pressure_performance": 0.5,
                "attendance_discipline": 0.5
            }
        
        # 1. Consistency: How stable are their scores?
        all_scores = []
        for t in self.previous_trimesters:
            for c in t.get("courses", []):
                if c.get("final") is not None:
                    all_scores.append(c["final"])
        
        consistency_score = 1.0 - (np.std(all_scores) / 25.0) if all_scores else 0.5
        consistency_score = np.clip(consistency_score, 0, 1)
        
        # 2. Improvement Rate: Are they getting better?
        trimester_avgs = [np.mean([c["final"] for c in t["courses"] if c.get("final") is not None]) 
                          for t in self.previous_trimesters]
        
        if len(trimester_avgs) >= 2:
            improvement_rate = (trimester_avgs[-1] - trimester_avgs[0]) / len(trimester_avgs)
        else:
            improvement_rate = 0.0
        
        # 3. Performance Under Pressure: Mid vs. Final exam comparison
        mid_scores = []
        final_scores = []
        for t in self.previous_trimesters:
            for c in t.get("courses", []):
                if c.get("mid") is not None and c.get("final") is not None:
                    mid_percentage = c["mid"] / 30.0  # Mid out of 30
                    # Estimate final exam portion from total final score
                    continuous = c.get("ct", 0) + c.get("assignment", 0) + c.get("attendance", 0) + c["mid"]
                    final_exam = c["final"] - continuous
                    final_percentage = final_exam / 40.0 if final_exam > 0 else 0  # Final exam out of 40
                    
                    mid_scores.append(mid_percentage)
                    final_scores.append(final_percentage)
        
        if mid_scores and final_scores:
            # Positive = performs better in finals, Negative = worse in finals
            pressure_performance = np.mean(final_scores) - np.mean(mid_scores)
            pressure_performance = np.clip(pressure_performance, -0.3, 0.3)  # Normalize to ±0.3
        else:
            pressure_performance = 0.0
        
        # 4. Attendance Discipline
        attendance_scores = []
        for t in self.previous_trimesters:
            for c in t.get("courses", []):
                if c.get("attendance") is not None:
                    max_attendance = 10 if "LAB" in c["course"] else 5
                    attendance_scores.append(c["attendance"] / max_attendance)
        
        attendance_discipline = np.mean(attendance_scores) if attendance_scores else 0.5
        
        return {
            "consistency_score": consistency_score,
            "improvement_rate": improvement_rate,
            "pressure_performance": pressure_performance,
            "attendance_discipline": attendance_discipline
        }
    
    def calculate_course_type_strengths(self) -> Dict:
        """Identify strengths in different course types"""
        
        if not self.previous_trimesters:
            return {
                "programming_strength": 0.0,
                "theory_strength": 0.0,
                "lab_strength": 0.0,
                "math_strength": 0.0
            }
        
        # Calculate averages for different course types
        prog_scores = []
        theory_scores = []
        lab_scores = []
        math_scores = []
        overall_scores = []
        
        for t in self.previous_trimesters:
            for c in t.get("courses", []):
                if c.get("final") is not None:
                    overall_scores.append(c["final"])
                    
                    if c["course"] in COURSE_CLUSTERS["PROG_CHAIN"]:
                        prog_scores.append(c["final"])
                    if c["course"] in COURSE_CLUSTERS["THEORY_SE"]:
                        theory_scores.append(c["final"])
                    if "LAB" in c["course"] or c["course"] in COURSE_CLUSTERS["LAB_CORE"]:
                        lab_scores.append(c["final"])
                    if c["course"] in COURSE_CLUSTERS["MATH"]:
                        math_scores.append(c["final"])
        
        overall_avg = np.mean(overall_scores) if overall_scores else 75.0
        
        # Strength = how much better than overall average
        programming_strength = (np.mean(prog_scores) - overall_avg) if prog_scores else 0.0
        theory_strength = (np.mean(theory_scores) - overall_avg) if theory_scores else 0.0
        lab_strength = (np.mean(lab_scores) - overall_avg) if lab_scores else 0.0
        math_strength = (np.mean(math_scores) - overall_avg) if math_scores else 0.0
        
        return {
            "programming_strength": programming_strength,
            "theory_strength": theory_strength,
            "lab_strength": lab_strength,
            "math_strength": math_strength
        }
    
    def calculate_behavioral_patterns(self) -> Dict:
        """Analyze behavioral patterns"""
        
        if not self.previous_trimesters:
            return {
                "ct_consistency": 0.5,
                "assignment_quality": 0.5,
                "mid_final_correlation": 0.0
            }
        
        ct_scores = []
        assignment_scores = []
        mid_scores = []
        final_scores = []
        
        for t in self.previous_trimesters:
            for c in t.get("courses", []):
                if c.get("ct") is not None:
                    ct_scores.append(c["ct"] / 20.0)
                if c.get("assignment") is not None:
                    max_assign = 10 if "LAB" not in c["course"] else 5
                    assignment_scores.append(c["assignment"] / max_assign)
                if c.get("mid") is not None:
                    mid_scores.append(c["mid"] / 30.0)
                if c.get("final") is not None:
                    final_scores.append(c["final"] / 100.0)
        
        ct_consistency = 1.0 - np.std(ct_scores) if len(ct_scores) > 1 else 0.5
        assignment_quality = np.mean(assignment_scores) if assignment_scores else 0.5
        
        # Correlation between mid and final performance
        if len(mid_scores) > 2 and len(final_scores) > 2:
            mid_final_correlation = np.corrcoef(mid_scores, final_scores)[0, 1]
        else:
            mid_final_correlation = 0.0
        
        return {
            "ct_consistency": ct_consistency,
            "assignment_quality": assignment_quality,
            "mid_final_correlation": mid_final_correlation
        }
    
    def apply_adjustments(self, base_prediction: float) -> Dict:
        """
        Apply meta-learning adjustments to base prediction
        
        Returns:
            Dict with adjusted prediction and adjustment breakdown
        """
        
        personal_metrics = self.calculate_personal_metrics()
        course_strengths = self.calculate_course_type_strengths()
        behavioral = self.calculate_behavioral_patterns()
        
        # Initialize adjustment
        total_adjustment = 0.0
        breakdown = {}
        
        # -------------------------
        # 1. Personal Performance Adjustment (±3 marks max)
        # -------------------------
        
        # Consistency bonus: Consistent students perform as expected
        if personal_metrics["consistency_score"] > 0.7:
            consistency_adj = 1.0
            breakdown["consistency_bonus"] = 1.0
        elif personal_metrics["consistency_score"] < 0.3:
            consistency_adj = -1.0
            breakdown["consistency_penalty"] = -1.0
        else:
            consistency_adj = 0.0
        
        total_adjustment += consistency_adj
        
        # Improvement trend adjustment
        improvement_adj = np.clip(personal_metrics["improvement_rate"], -2, 2)
        if abs(improvement_adj) > 0.5:
            breakdown["improvement_trend"] = round(improvement_adj, 2)
            total_adjustment += improvement_adj
        
        # -------------------------
        # 2. Course-Type Strength Adjustment (±4 marks max)
        # -------------------------
        
        course_type_adj = 0.0
        
        # Identify current course type
        if self.course_code in COURSE_CLUSTERS["PROG_CHAIN"]:
            course_type_adj = np.clip(course_strengths["programming_strength"] * 0.5, -3, 3)
            if abs(course_type_adj) > 0.5:
                breakdown["programming_strength"] = round(course_type_adj, 2)
        
        elif "LAB" in self.course_code or self.course_code in COURSE_CLUSTERS["LAB_CORE"]:
            course_type_adj = np.clip(course_strengths["lab_strength"] * 0.5, -3, 3)
            if abs(course_type_adj) > 0.5:
                breakdown["lab_strength"] = round(course_type_adj, 2)
        
        elif self.course_code in COURSE_CLUSTERS["MATH"]:
            course_type_adj = np.clip(course_strengths["math_strength"] * 0.5, -3, 3)
            if abs(course_type_adj) > 0.5:
                breakdown["math_strength"] = round(course_type_adj, 2)
        
        elif self.course_code in COURSE_CLUSTERS["THEORY_SE"]:
            course_type_adj = np.clip(course_strengths["theory_strength"] * 0.5, -3, 3)
            if abs(course_type_adj) > 0.5:
                breakdown["theory_strength"] = round(course_type_adj, 2)
        
        total_adjustment += course_type_adj
        
        # -------------------------
        # 3. Behavioral Pattern Adjustment (±2 marks max)
        # -------------------------
        
        # Current performance vs. historical patterns
        current_ct_perf = self.current_course.get("ct", 0) / 20.0
        current_mid_perf = self.current_course.get("mid", 0) / 30.0
        
        # If student is performing better in current semester CTs
        if behavioral["ct_consistency"] > 0.6 and current_ct_perf > behavioral["ct_consistency"]:
            behavioral_adj = (current_ct_perf - behavioral["ct_consistency"]) * 5
            behavioral_adj = np.clip(behavioral_adj, 0, 1.5)
            if behavioral_adj > 0.3:
                breakdown["current_improvement"] = round(behavioral_adj, 2)
                total_adjustment += behavioral_adj
        
        # Pressure performance: Adjust based on mid-to-final pattern
        pressure_adj = personal_metrics["pressure_performance"] * 3  # Scale to ±0.9
        if abs(pressure_adj) > 0.3:
            breakdown["pressure_performance"] = round(pressure_adj, 2)
            total_adjustment += pressure_adj
        
        # -------------------------
        # Cap total adjustment to ±5 marks
        # -------------------------
        total_adjustment = np.clip(total_adjustment, -5, 5)
        
        adjusted_prediction = base_prediction + total_adjustment
        adjusted_prediction = np.clip(adjusted_prediction, 0, 100)
        
        return {
            "base_prediction": round(base_prediction, 2),
            "adjustment": round(total_adjustment, 2),
            "adjusted_prediction": round(adjusted_prediction, 2),
            "adjustment_breakdown": breakdown,
            "personal_metrics": {k: round(v, 3) for k, v in personal_metrics.items()},
            "course_strengths": {k: round(v, 2) for k, v in course_strengths.items()},
            "behavioral_patterns": {k: round(v, 3) for k, v in behavioral.items()}
        }


# -------------------------
# MAIN PREDICTION API
# -------------------------

def predict_grade(
    current_course: Dict,
    previous_trimesters: Optional[List[Dict]] = None,
    cgpa: Optional[float] = None,
    current_trimester: int = 1,
    use_meta_learning: bool = True
) -> Dict:
    """
    Main API function to predict student grade
    
    Args:
        current_course: Dict with current course marks
            {
                "course": "CSE1110",
                "ct": 16,
                "assignment": 8,
                "attendance": 5,
                "mid": 24,
                "project": 0  # Optional, for courses with projects
            }
        
        previous_trimesters: List of previous trimester data (None for fresh students)
            [
                {
                    "trimester": 1,
                    "gpa": 3.2,
                    "courses": [
                        {
                            "course": "CSE1110",
                            "ct": 18,
                            "assignment": 7,
                            "attendance": 5,
                            "mid": 25,
                            "final": 33,
                            "grade": "B",
                            "grade_point": 3.0
                        }
                    ]
                }
            ]
        
        cgpa: Current CGPA (optional, will be estimated if not provided)
        current_trimester: Current trimester number
        use_meta_learning: Apply Layer 2 personalized adjustments (only for experienced students)
    
    Returns:
        Dict with prediction results
    """
    
    # Load models
    models = ModelLoader()
    
    # Build feature vector (Layer 1)
    features = build_feature_vector(current_course, previous_trimesters, cgpa, current_trimester)
    
    # Get base prediction
    base_result = predict_with_hybrid(features, models)
    
    # ✅ NEW: Calculate mathematical ceiling
    max_possible = calculate_max_possible_score(current_course)
    
    # Cap base prediction if it exceeds mathematical possibility
    if base_result["predicted_score"] > max_possible:
        base_result["predicted_score"] = max_possible
        base_result["model_used"] += " (Capped)"
        base_result["predicted_grade"], base_result["grade_point"] = score_to_grade(max_possible)
    
    # Apply Layer 2 personalization for experienced students
    if use_meta_learning and previous_trimesters and len(previous_trimesters) >= 1:
        adjuster = PersonalizedAdjuster(previous_trimesters, current_course)
        meta_result = adjuster.apply_adjustments(base_result["predicted_score"])
        
        # ✅ NEW: Cap adjusted prediction
        if meta_result["adjusted_prediction"] > max_possible:
            meta_result["adjusted_prediction"] = max_possible
        
        final_grade, final_gp = score_to_grade(meta_result["adjusted_prediction"])
        
        return {
            "course": current_course.get("course", "Unknown"),
            "student_type": "Experienced" if len(previous_trimesters) >= 3 else "Intermediate",
            
            # Layer 1: Base Model Prediction
            "layer1_prediction": {
                "score": base_result["predicted_score"],
                "grade": base_result["predicted_grade"],
                "grade_point": base_result["grade_point"],
                "model_used": base_result["model_used"],
                "confidence": base_result["confidence"]
            },
            
            # Layer 2: Personalized Adjustment
            "layer2_adjustment": {
                "adjustment_amount": meta_result["adjustment"],
                "adjustment_breakdown": meta_result["adjustment_breakdown"],
                "personal_insights": meta_result["personal_metrics"],
                "course_type_strengths": meta_result["course_strengths"],
                "behavioral_patterns": meta_result["behavioral_patterns"]
            },
            
            # Final Prediction
            "final_prediction": {
                "score": meta_result["adjusted_prediction"],
                "grade": final_grade,
                "grade_point": final_gp,
                "max_possible_score": max_possible
            },
            
            # Additional info
            "current_marks": {
                "ct": current_course.get("ct", 0),
                "assignment": current_course.get("assignment", 0),
                "attendance": current_course.get("attendance", 0),
                "mid": current_course.get("mid", 0),
                "total_earned": int(features["curr_total"].values[0]),
                "percentage": round(float(features["curr_percentage"].values[0]), 2)
            }
        }
    
    else:
        # Fresh student or meta-learning disabled
        return {
            "course": current_course.get("course", "Unknown"),
            "student_type": "Fresh",
            
            # Only Layer 1 prediction
            "layer1_prediction": {
                "score": base_result["predicted_score"],
                "grade": base_result["predicted_grade"],
                "grade_point": base_result["grade_point"],
                "model_used": base_result["model_used"],
                "confidence": base_result["confidence"]
            },
            
            "layer2_adjustment": {
                "enabled": False,
                "reason": "Insufficient historical data (requires ≥1 trimester)"
            },
            
            # Final Prediction (same as Layer 1 for fresh students)
            "final_prediction": {
                "score": base_result["predicted_score"],
                "grade": base_result["predicted_grade"],
                "grade_point": base_result["grade_point"],
                "max_possible_score": max_possible
            },
            
            "current_marks": {
                "ct": current_course.get("ct", 0),
                "assignment": current_course.get("assignment", 0),
                "attendance": current_course.get("attendance", 0),
                "mid": current_course.get("mid", 0),
                "total_earned": int(features["curr_total"].values[0]),
                "percentage": round(float(features["curr_percentage"].values[0]), 2)
            }
        }


# -------------------------
# EXAMPLE USAGE
# -------------------------

if __name__ == "__main__":
    
    print("="*80)
    print("COSMOS-ITS GRADE PREDICTION API - EXAMPLES")
    print("="*80)
    
    # -------------------------
    # EXAMPLE 1: Fresh Student (1st Trimester)
    # -------------------------
    print("\n📘 EXAMPLE 1: First Trimester Student (CSE1110 - ICS)")
    print("-" * 80)
    
    example1_input = {
        "course": "CSE1110",
        "ct": 16,
        "assignment": 8,
        "attendance": 5,
        "mid": 24
    }
    
    result1 = predict_grade(
        current_course=example1_input,
        previous_trimesters=None,  # No previous data
        current_trimester=1
    )
    
    print(json.dumps(result1, indent=2))
    
    # -------------------------
    # EXAMPLE 2: Experienced Student (4th Trimester)
    # -------------------------
    print("\n\n📗 EXAMPLE 2: Fourth Trimester Student (CSE2118 - AOOP)")
    print("-" * 80)
    
    example2_input = {
        "course": "CSE2118",
        "ct": 16,
        "assignment": 10,
        "attendance": 10,
        "mid": 20,
        "project": 15
    }
    
    example2_history = [
        {
            "trimester": 1,
            "gpa": 3.2,
            "courses": [
                {
                    "course": "CSE1110",
                    "ct": 18,
                    "assignment": 7,
                    "attendance": 5,
                    "mid": 25,
                    "final": 75,
                    "grade": "B-",
                    "grade_point": 2.67
                },
                {
                    "course": "ENG1011",
                    "ct": 15,
                    "assignment": 8,
                    "attendance": 4,
                    "mid": 22,
                    "final": 70,
                    "grade": "C+",
                    "grade_point": 2.33
                }
            ]
        },
        {
            "trimester": 2,
            "gpa": 3.4,
            "courses": [
                {
                    "course": "CSE1111",
                    "ct": 17,
                    "assignment": 9,
                    "attendance": 5,
                    "mid": 26,
                    "final": 80,
                    "grade": "B",
                    "grade_point": 3.0
                },
                {
                    "course": "CSE1112",
                    "ct": 18,
                    "assignment": 5,
                    "attendance": 10,
                    "mid": 27,
                    "final": 85,
                    "grade": "B+",
                    "grade_point": 3.33
                }
            ]
        },
        {
            "trimester": 3,
            "gpa": 3.5,
            "courses": [
                {
                    "course": "CSE1115",
                    "ct": 18,
                    "assignment": 9,
                    "attendance": 5,
                    "mid": 27,
                    "final": 83,
                    "grade": "B+",
                    "grade_point": 3.33
                },
                {
                    "course": "CSE1116",
                    "ct": 19,
                    "assignment": 5,
                    "attendance": 10,
                    "mid": 28,
                    "final": 88,
                    "grade": "A-",
                    "grade_point": 3.67
                }
            ]
        }
    ]
    
    result2 = predict_grade(
        current_course=example2_input,
        previous_trimesters=example2_history,
        cgpa=3.37,  # Average of 3.2, 3.4, 3.5
        current_trimester=4,
        use_meta_learning=True  # Enable Layer 2 adjustments
    )
    
    print(json.dumps(result2, indent=2))
    
    print("\n" + "="*80)
    print("✅ API READY FOR INTEGRATION WITH YOUR WEBSITE!")
    print("="*80)
