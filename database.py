# ============================================
# Database Manager (Supabase)
# ============================================
# File: database.py
# Mô tả: Kết nối và thao tác với Supabase PostgreSQL
# ============================================

import os
import json
from datetime import datetime
from typing import List, Optional, Dict

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("⚠️ supabase-py chưa cài. Chạy: pip install supabase")


class DatabaseManager:
    """
    Manager class cho Supabase database.
    
    Chức năng:
    - Kết nối Supabase
    - Lưu application & kết quả predict
    - Truy vấn lịch sử applications
    
    Lưu ý:
    - Nếu không có env vars (SUPABASE_URL, SUPABASE_KEY), 
      API vẫn chạy bình thường nhưng không lưu data
    - Đây là thiết kế "graceful degradation"
    """
    
    TABLE_NAME = "applications"  # Tên table trong Supabase
    
    def __init__(self):
        """Khởi tạo database connection."""
        self.client: Optional[Client] = None
        self._connect()
    
    def _connect(self):
        """Kết nối tới Supabase."""
        if not SUPABASE_AVAILABLE:
            print("⚠️ Supabase library not available. Database features disabled.")
            return
        
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            print("⚠️ SUPABASE_URL hoặc SUPABASE_KEY chưa được set.")
            print("   → Database features sẽ bị disabled.")
            print("   → Thêm vào file .env hoặc Render environment variables.")
            return
        
        try:
            self.client = create_client(supabase_url, supabase_key)
            print(f"✅ Connected to Supabase: {supabase_url[:40]}...")
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            self.client = None
    
    def is_connected(self) -> bool:
        """Kiểm tra đã kết nối database chưa."""
        return self.client is not None
    
    def save_application(self, application_data, prediction_result: dict) -> Optional[str]:
        """
        Lưu application và kết quả predict vào database.
        
        Args:
            application_data: Pydantic model ApplicationInput
            prediction_result: Dict kết quả predict
            
        Returns:
            record_id (str) hoặc None nếu lỗi
        """
        if not self.is_connected():
            return None
        
        try:
            record = {
                "input_data": application_data.dict(),
                "approval_score": prediction_result["approval_score"],
                "approved": prediction_result["approved"],
                "risk_level": prediction_result["risk_level"],
                "recommendation": prediction_result["recommendation"],
                "created_at": datetime.utcnow().isoformat()
            }
            
            response = (
                self.client
                .table(self.TABLE_NAME)
                .insert(record)
                .execute()
            )
            
            if response.data and len(response.data) > 0:
                record_id = str(response.data[0].get("id", ""))
                print(f"✅ Saved application: {record_id}")
                return record_id
            
            return None
        
        except Exception as e:
            print(f"❌ Error saving application: {e}")
            raise
    
    def get_applications(self, limit: int = 20, offset: int = 0) -> List[Dict]:
        """
        Lấy danh sách applications (mới nhất trước).
        
        Args:
            limit: Số record tối đa
            offset: Bỏ qua bao nhiêu record đầu
            
        Returns:
            List of application records
        """
        if not self.is_connected():
            return []
        
        try:
            response = (
                self.client
                .table(self.TABLE_NAME)
                .select("*")
                .order("created_at", desc=True)
                .range(offset, offset + limit - 1)
                .execute()
            )
            
            return response.data if response.data else []
        
        except Exception as e:
            print(f"❌ Error fetching applications: {e}")
            raise
    
    def get_application_by_id(self, application_id: str) -> Optional[Dict]:
        """
        Lấy chi tiết một application theo ID.
        
        Args:
            application_id: UUID của record
            
        Returns:
            Application record hoặc None
        """
        if not self.is_connected():
            return None
        
        try:
            response = (
                self.client
                .table(self.TABLE_NAME)
                .select("*")
                .eq("id", application_id)
                .execute()
            )
            
            if response.data and len(response.data) > 0:
                return response.data[0]
            
            return None
        
        except Exception as e:
            print(f"❌ Error fetching application {application_id}: {e}")
            raise
