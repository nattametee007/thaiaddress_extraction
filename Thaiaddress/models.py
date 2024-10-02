
from sqlalchemy import Column, String, JSON, ForeignKey
from database import Base
import json
from sqlalchemy.dialects.postgresql import UUID
import uuid
from sqlalchemy.orm import relationship

class Extraction(Base):
    __tablename__ = "extractions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)  # Ensure UUID type for id
    app_name = Column(String, nullable=False)
    app_metadata = Column(String, nullable=False)
    extraction_data = Column(String, nullable=False)
    @property
    def app_metadata_parse_data(self):
        return json.loads(self.app_metadata)
    
    @property
    def extraction_parse_data(self):
        return json.loads(self.extraction_data)
    
# Transaction table
class Transaction(Base):
    __tablename__ = "transactions"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    main_id = Column(UUID(as_uuid=True), ForeignKey('extractions.id'), nullable=False)
    service_name = Column(String, nullable=False)
    response = Column(String, nullable=False)  # Store the response as a JSON string
    feedback = Column(String, default=None)
    status = Column(String, default='0')  # Default status
    # Relationship to Extraction
    extraction = relationship("Extraction", back_populates="transactions")
    @property
    def transactions_parse_data(self):
        """Parses the response JSON string into a Python dictionary."""
        return json.loads(self.response)


# Add relationship to Extraction model
Extraction.transactions = relationship("Transaction", back_populates="extraction")