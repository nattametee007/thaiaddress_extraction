from fastapi import Depends, FastAPI, HTTPException, Header, APIRouter
import uuid
from sqlalchemy.orm import Session
import thaiaddress
from sqlalchemy.exc import SQLAlchemyError
from database import Base, engine, get_db
from schema import ExtractionInput, ExtractionResponse, TransactionResponse, UpdateTransactionRequest, AppMetadata
from models import Extraction, Transaction
import json

app = FastAPI()
Base.metadata.create_all(bind=engine)

def parse_address(text: str, attributes: list) -> dict:
    try:
        return thaiaddress.parse(text, fields=attributes)
    except Exception as e:
        print(f"Error parsing address: {e}")
        return {}

def clean_response(response_data: dict, service_name: str) -> dict:
    if service_name == 'address':
        return response_data.get('data', {})
    elif service_name == 'phone':
        return {'phone_numbers': response_data.get('data', [])}
    elif service_name == 'email':
        return {'email_addresses': response_data.get('data', [])}
    else:
        return response_data 
    
@app.post("/extractions/", response_model=ExtractionResponse)
async def create_item(
    item: ExtractionInput,
    db: Session = Depends(get_db),
    app_name: str = Header(..., description="Application Name"),
    app_metadata: str = Header(
        ...,
        description="App Metadata (JSON-formatted string)",
        example='{"business_id": "13123213", "contact_id": "67890", "custom_field": "value"}'
    )
):
    if app_name != 'oho-app':
        raise HTTPException(status_code=403, detail="Unauthorized application")

    if not set(item.attributes).issubset({'phone', 'address', 'email'}) or not item.attributes:
        raise HTTPException(status_code=400, detail="Invalid attributes. Must be 'address', 'phone', or 'email'")

    try:
        metadata_dict = json.loads(app_metadata)
        app_metadata_model = AppMetadata(root=metadata_dict)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in app_metadata: {str(e)}")

    extraction_data = parse_address(item.text, item.attributes)
    
    if extraction_data:
        extract_uuid = str(uuid.uuid4())

        try:
            extraction_data_json = json.dumps(extraction_data, ensure_ascii=False)
            db_extraction = Extraction(
                id=extract_uuid,
                app_name=app_name,
                app_metadata=json.dumps(app_metadata_model.root, ensure_ascii=False),
                extraction_data=extraction_data_json
            )
            db.add(db_extraction)
            
            for field in ['address', 'phone', 'email']:
                if field in extraction_data:
                    transaction_uuid = str(uuid.uuid4())
                    response_data = extraction_data[field]
                    
                    clean_response_data = clean_response(response_data, field)
                    
                    db_transaction = Transaction(
                        id=transaction_uuid,
                        main_id=extract_uuid,
                        service_name=field,
                        response=json.dumps(clean_response_data, ensure_ascii=False),
                        status='0'
                    )
                    db.add(db_transaction)
                    
            db.commit()

        except SQLAlchemyError as e:
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

        return ExtractionResponse(extraction_data=extraction_data, id=extract_uuid, attributes=item.attributes)
    
    else:
        return ExtractionResponse(extraction_data={}, id=None, attributes=item.attributes)

@app.put("/extractions/feedback/{extraction_id}", response_model=TransactionResponse)
async def update_transaction(
    extraction_id: str,
    request: UpdateTransactionRequest,
    db: Session = Depends(get_db)
):
    try:
        extraction = db.query(Extraction).filter(Extraction.id == extraction_id).first()
        if not extraction:
            raise HTTPException(status_code=404, detail="Extraction not found")

        transactions = db.query(Transaction).filter(Transaction.main_id == extraction_id).all()
        if not transactions:
            raise HTTPException(status_code=404, detail="No transactions found for this extraction")

        updated_transactions = []

        for transaction in transactions:
            service_name = transaction.service_name

            if service_name in request.feedback:
                service_feedback = request.feedback[service_name]
                clean_feedback = service_feedback.edit_data

                transaction.status = '1' if service_feedback.edit_details else '2'
                transaction.feedback = json.dumps(clean_feedback, ensure_ascii=False)
            else:
                transaction.status = '0'
                transaction.feedback = '{}'

            updated_transactions.append({
                "id": transaction.id,
                "service_name": transaction.service_name,
                "status": transaction.status,
                "feedback": json.loads(transaction.feedback) if transaction.feedback else None
            })

        db.commit()

        return TransactionResponse(
            extraction_id=extraction_id,
            transactions=updated_transactions
        )

    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")