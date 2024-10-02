
from pydantic import BaseModel,Field,RootModel
from typing import List, Optional, Dict,Any

class ExtractionInput(BaseModel):
    text: str
    attributes: List[str] | List[None] 

class ExtractionResponse(BaseModel):
    extraction_data: dict
    id: str | None
    attributes: List[str] | List[None] 

class ServiceFeedback(BaseModel):
    edit_data: Dict[str, Any] = Field(default_factory=dict)
    edit_details: bool = False

class UpdateTransactionRequest(BaseModel):
    feedback: Dict[str, ServiceFeedback] = Field(default_factory=dict)

class TransactionResponse(BaseModel):
    extraction_id: str
    transactions: List[Dict[str, Any]]

class AppMetadata(RootModel):
    root: Dict[str, Any]