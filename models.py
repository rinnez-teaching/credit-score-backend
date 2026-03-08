# ============================================
# Pydantic Models (Request/Response Schemas)
# ============================================
# File: models.py
# Mô tả: Định nghĩa cấu trúc dữ liệu input/output cho API
# ============================================

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict
from datetime import datetime


# ============================================
# REQUEST MODELS (Input)
# ============================================

class ApplicationInput(BaseModel):
    """
    Schema cho input hồ sơ vay.
    Mỗi field có validation (min, max) và mô tả rõ ràng.
    """
    
    income: float = Field(
        ...,  # ... = required
        ge=0, le=500000,
        description="Thu nhập hàng năm (USD)",
        examples=[75000]
    )
    
    age: int = Field(
        ...,
        ge=18, le=80,
        description="Tuổi của người nộp đơn",
        examples=[35]
    )
    
    employment_years: int = Field(
        ...,
        ge=0, le=50,
        description="Số năm đã làm việc",
        examples=[8]
    )
    
    loan_amount: float = Field(
        ...,
        ge=0, le=500000,
        description="Số tiền muốn vay (USD)",
        examples=[25000]
    )
    
    loan_term: int = Field(
        ...,
        ge=6, le=120,
        description="Thời hạn vay (tháng): 6, 12, 24, 36, 48, 60",
        examples=[36]
    )
    
    credit_history_length: int = Field(
        ...,
        ge=0, le=50,
        description="Độ dài lịch sử tín dụng (năm)",
        examples=[12]
    )
    
    num_credit_lines: int = Field(
        ...,
        ge=0, le=20,
        description="Số dòng tín dụng đang hoạt động",
        examples=[4]
    )
    
    num_delinquencies: int = Field(
        ...,
        ge=0, le=10,
        description="Số lần trễ hạn thanh toán trong quá khứ",
        examples=[0]
    )
    
    debt_to_income_ratio: float = Field(
        ...,
        ge=0.0, le=1.0,
        description="Tỷ lệ nợ/thu nhập (0.0 - 1.0)",
        examples=[0.28]
    )
    
    savings_balance: float = Field(
        ...,
        ge=0, le=1000000,
        description="Số dư tài khoản tiết kiệm (USD)",
        examples=[10000]
    )
    
    property_value: float = Field(
        ...,
        ge=0, le=2000000,
        description="Giá trị tài sản sở hữu (0 nếu không có) (USD)",
        examples=[300000]
    )
    
    education_level: int = Field(
        ...,
        ge=1, le=4,
        description="Trình độ học vấn: 1=THPT, 2=Đại học, 3=Thạc sĩ, 4=Tiến sĩ",
        examples=[3]
    )
    
    employment_type: int = Field(
        ...,
        ge=1, le=3,
        description="Loại hình công việc: 1=Toàn thời gian, 2=Bán thời gian, 3=Tự kinh doanh",
        examples=[1]
    )
    
    # Custom validation
    @validator('loan_amount')
    def loan_amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Số tiền vay phải lớn hơn 0')
        return v
    
    @validator('income')
    def income_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Thu nhập phải lớn hơn 0')
        return v
    
    class Config:
        json_schema_extra = {
            "example": {
                "income": 75000,
                "age": 35,
                "employment_years": 8,
                "loan_amount": 25000,
                "loan_term": 36,
                "credit_history_length": 12,
                "num_credit_lines": 4,
                "num_delinquencies": 0,
                "debt_to_income_ratio": 0.28,
                "savings_balance": 10000,
                "property_value": 300000,
                "education_level": 3,
                "employment_type": 1
            }
        }


# ============================================
# RESPONSE MODELS (Output)
# ============================================

class PredictionResponse(BaseModel):
    """Schema cho kết quả predict."""
    
    approval_score: float = Field(
        ...,
        description="Điểm phê duyệt (0-100%)"
    )
    approved: bool = Field(
        ...,
        description="Kết quả phê duyệt: True = Approved, False = Rejected"
    )
    risk_level: str = Field(
        ...,
        description="Mức rủi ro: Low, Medium, High"
    )
    recommendation: str = Field(
        ...,
        description="Khuyến nghị cho hồ sơ"
    )
    record_id: Optional[str] = Field(
        None,
        description="ID record trong database (nếu lưu thành công)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "approval_score": 85.5,
                "approved": True,
                "risk_level": "Low",
                "recommendation": "Hồ sơ có điểm tín dụng tốt. Khuyến nghị phê duyệt.",
                "record_id": "abc-123"
            }
        }


class HealthResponse(BaseModel):
    """Schema cho health check response."""
    
    status: str
    model_loaded: bool
    database_connected: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Schema cho thông tin model."""
    
    model_type: str
    feature_count: int
    feature_names: List[str]
    feature_descriptions: Dict
    model_metrics: Dict


class ApplicationRecord(BaseModel):
    """Schema cho record lịch sử application."""
    
    id: str
    input_data: Dict
    approval_score: float
    approved: bool
    risk_level: str
    recommendation: str
    created_at: str
