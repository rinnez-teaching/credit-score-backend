# ============================================
# Backend: FastAPI + XGBoost Credit Score API
# ============================================
# File: main.py
# Mô tả: Entry point cho FastAPI application
# ============================================

import os
from datetime import datetime
from typing import List, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import ApplicationInput, PredictionResponse, ApplicationRecord, HealthResponse, ModelInfoResponse
from ml_model import CreditScoreModel
from database import DatabaseManager

# ============================================
# KHỞI TẠO APP
# ============================================

app = FastAPI(
    title="Credit Score Prediction API",
    description="""
    🏦 API dự đoán điểm tín dụng (Credit Score) sử dụng XGBoost.
    
    ## Chức năng
    - **Predict**: Nhận thông tin hồ sơ vay → Trả về approval score
    - **History**: Xem lịch sử các hồ sơ đã submit
    - **Model Info**: Xem thông tin model đang sử dụng
    
    ## Tech Stack
    - FastAPI (Python)
    - XGBoost (ML Model)
    - Supabase (Database)
    """,
    version="1.0.0",
    docs_url="/docs",       # Swagger UI
    redoc_url="/redoc",     # ReDoc UI
)

# ============================================
# CORS MIDDLEWARE
# ============================================
# Cho phép frontend (Vercel) gọi API
# Trong production, thay "*" bằng domain frontend cụ thể

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# KHỞI TẠO MODEL & DATABASE
# ============================================

# Load ML model (chỉ load 1 lần khi start server)
MODEL_PATH = os.getenv("MODEL_PATH", "xgboost_model.json")
FEATURE_PATH = os.getenv("FEATURE_PATH", "feature_names.json")

credit_model = CreditScoreModel(MODEL_PATH, FEATURE_PATH)

# Khởi tạo database connection
db = DatabaseManager()


# ============================================
# API ENDPOINTS
# ============================================

@app.get("/", tags=["General"])
async def root():
    """Welcome endpoint - kiểm tra API hoạt động."""
    return {
        "message": "🏦 Credit Score Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint.
    Kiểm tra server, model, và database có hoạt động không.
    """
    model_loaded = credit_model.is_loaded()
    db_connected = db.is_connected()
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        database_connected=db_connected,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Trả về thông tin model đang sử dụng.
    Bao gồm: feature names, metrics, v.v.
    """
    if not credit_model.is_loaded():
        raise HTTPException(status_code=503, detail="Model chưa được load")
    
    return credit_model.get_model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_credit_score(application: ApplicationInput):
    """
    🎯 Dự đoán Credit Score cho một hồ sơ vay.
    
    **Input**: Thông tin hồ sơ vay (income, age, loan_amount, v.v.)
    
    **Output**: 
    - `approval_score`: Điểm phê duyệt (0-100%)
    - `approved`: True/False
    - `recommendation`: Khuyến nghị
    - `risk_level`: Mức rủi ro (Low/Medium/High)
    """
    if not credit_model.is_loaded():
        raise HTTPException(status_code=503, detail="Model chưa được load. Vui lòng thử lại sau.")
    
    try:
        # 1. Predict bằng ML model
        result = credit_model.predict(application)
        
        # 2. Lưu vào database (nếu có kết nối)
        if db.is_connected():
            try:
                record_id = db.save_application(application, result)
                result["record_id"] = record_id
            except Exception as e:
                # Nếu lỗi DB, vẫn trả kết quả predict (không block user)
                print(f"⚠️ Database error (non-blocking): {e}")
                result["record_id"] = None
        else:
            result["record_id"] = None
        
        return PredictionResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Input không hợp lệ: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi server: {str(e)}")


@app.get("/applications", response_model=List[ApplicationRecord], tags=["History"])
async def get_applications(limit: int = 20, offset: int = 0):
    """
    Lấy danh sách lịch sử các hồ sơ đã submit.
    Hỗ trợ pagination (limit + offset).
    """
    if not db.is_connected():
        raise HTTPException(status_code=503, detail="Database chưa kết nối")
    
    try:
        records = db.get_applications(limit=limit, offset=offset)
        return records
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi database: {str(e)}")


@app.get("/applications/{application_id}", response_model=ApplicationRecord, tags=["History"])
async def get_application_detail(application_id: str):
    """
    Lấy chi tiết một hồ sơ theo ID.
    """
    if not db.is_connected():
        raise HTTPException(status_code=503, detail="Database chưa kết nối")
    
    try:
        record = db.get_application_by_id(application_id)
        if not record:
            raise HTTPException(status_code=404, detail="Không tìm thấy hồ sơ")
        return record
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi database: {str(e)}")


# ============================================
# STARTUP EVENT
# ============================================

@app.on_event("startup")
async def startup_event():
    """Chạy khi server khởi động."""
    print("=" * 60)
    print("🚀 Credit Score Prediction API - Starting...")
    print(f"📦 Model: {MODEL_PATH}")
    print(f"📋 Features: {FEATURE_PATH}")
    print(f"🔗 Database: {'Connected' if db.is_connected() else 'Not connected'}")
    print(f"🌐 CORS Origins: {ALLOWED_ORIGINS}")
    print("=" * 60)
