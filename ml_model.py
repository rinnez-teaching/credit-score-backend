# ============================================
# ML Model Manager (XGBoost)
# ============================================
# File: ml_model.py
# Mô tả: Load model XGBoost, predict credit score
# ============================================

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb


class CreditScoreModel:
    """
    Manager class cho XGBoost Credit Score Model.
    
    Chức năng:
    - Load model từ file .json
    - Load feature metadata
    - Predict credit score từ input data
    - Trả về kết quả kèm risk level & recommendation
    """
    
    def __init__(self, model_path: str, feature_path: str):
        """
        Khởi tạo model manager.
        
        Args:
            model_path: Đường dẫn tới file model (.json)
            feature_path: Đường dẫn tới file feature_names.json
        """
        self.model = None
        self.feature_info = None
        self.feature_names = None
        
        # Load model
        self._load_model(model_path)
        
        # Load feature metadata
        self._load_feature_info(feature_path)
    
    def _load_model(self, model_path: str):
        """Load XGBoost model từ file."""
        try:
            if not os.path.exists(model_path):
                print(f"⚠️ Model file not found: {model_path}")
                print("   → API sẽ chạy nhưng endpoint /predict sẽ trả lỗi 503")
                return
            
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            print(f"✅ Model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def _load_feature_info(self, feature_path: str):
        """Load feature metadata từ file JSON."""
        try:
            if not os.path.exists(feature_path):
                print(f"⚠️ Feature file not found: {feature_path}")
                # Fallback: dùng default feature names
                self.feature_names = [
                    'income', 'age', 'employment_years', 'loan_amount',
                    'loan_term', 'credit_history_length', 'num_credit_lines',
                    'num_delinquencies', 'debt_to_income_ratio', 'savings_balance',
                    'property_value', 'education_level', 'employment_type'
                ]
                self.feature_info = {"feature_names": self.feature_names}
                return
            
            with open(feature_path, 'r') as f:
                self.feature_info = json.load(f)
            
            self.feature_names = self.feature_info.get("feature_names", [])
            print(f"✅ Feature info loaded: {feature_path} ({len(self.feature_names)} features)")
        except Exception as e:
            print(f"❌ Error loading feature info: {e}")
    
    def is_loaded(self) -> bool:
        """Kiểm tra model đã load thành công chưa."""
        return self.model is not None
    
    def get_model_info(self) -> dict:
        """Trả về thông tin model."""
        return {
            "model_type": "XGBoost Classifier",
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
            "feature_descriptions": self.feature_info.get("feature_descriptions", {}),
            "model_metrics": self.feature_info.get("model_metrics", {})
        }
    
    def predict(self, application_data) -> dict:
        """
        Predict credit score cho một hồ sơ vay.
        
        Args:
            application_data: Pydantic model ApplicationInput
            
        Returns:
            dict với keys: approval_score, approved, risk_level, recommendation
        """
        if not self.is_loaded():
            raise RuntimeError("Model chưa được load")
        
        # 1. Chuyển input thành DataFrame (đúng thứ tự features)
        input_dict = application_data.dict()
        input_df = pd.DataFrame([{
            feature: input_dict[feature] 
            for feature in self.feature_names
        }])
        
        # 2. Predict probability
        probabilities = self.model.predict_proba(input_df)[0]
        approval_probability = float(probabilities[1])  # Probability của class 1 (approved)
        approval_score = round(approval_probability * 100, 2)  # Chuyển sang %
        
        # 3. Quyết định approved/rejected (threshold = 50%)
        approved = approval_probability >= 0.5
        
        # 4. Xác định risk level
        risk_level = self._get_risk_level(approval_probability)
        
        # 5. Tạo recommendation
        recommendation = self._get_recommendation(
            approval_probability, risk_level, input_dict
        )
        
        return {
            "approval_score": approval_score,
            "approved": approved,
            "risk_level": risk_level,
            "recommendation": recommendation
        }
    
    def _get_risk_level(self, probability: float) -> str:
        """
        Xác định mức rủi ro dựa trên probability.
        
        - >= 0.75: Low Risk (rủi ro thấp, nên phê duyệt)
        - 0.40 - 0.75: Medium Risk (cần xem xét kỹ)
        - < 0.40: High Risk (rủi ro cao, nên từ chối)
        """
        if probability >= 0.75:
            return "Low"
        elif probability >= 0.40:
            return "Medium"
        else:
            return "High"
    
    def _get_recommendation(self, probability: float, risk_level: str, input_data: dict) -> str:
        """
        Tạo recommendation text dựa trên kết quả predict và input data.
        """
        score_pct = probability * 100
        
        if risk_level == "Low":
            base = f"✅ Hồ sơ có điểm tín dụng tốt ({score_pct:.1f}%). Khuyến nghị PHÊ DUYỆT."
        elif risk_level == "Medium":
            base = f"⚠️ Hồ sơ ở mức rủi ro trung bình ({score_pct:.1f}%). Cần xem xét thêm."
        else:
            base = f"❌ Hồ sơ có rủi ro cao ({score_pct:.1f}%). Khuyến nghị TỪ CHỐI hoặc yêu cầu thêm thông tin."
        
        # Thêm gợi ý cụ thể dựa trên input
        suggestions = []
        
        if input_data.get("debt_to_income_ratio", 0) > 0.4:
            suggestions.append("Tỷ lệ nợ/thu nhập cao (>{:.0%}). Nên giảm nợ trước khi vay.".format(
                input_data["debt_to_income_ratio"]
            ))
        
        if input_data.get("num_delinquencies", 0) > 0:
            suggestions.append(f"Có {input_data['num_delinquencies']} lần trễ hạn thanh toán. Ảnh hưởng tiêu cực.")
        
        if input_data.get("employment_years", 0) < 2:
            suggestions.append("Kinh nghiệm làm việc dưới 2 năm. Nên có thêm người bảo lãnh.")
        
        loan = input_data.get("loan_amount", 0)
        income = input_data.get("income", 1)
        if loan > income * 0.5:
            suggestions.append("Số tiền vay vượt 50% thu nhập năm. Nên giảm số tiền vay.")
        
        if suggestions:
            base += "\n\n💡 Gợi ý:\n" + "\n".join(f"  - {s}" for s in suggestions)
        
        return base
