from fastapi import APIRouter, HTTPException
from controllers import InferenceController, LoadModel
from pydantic import BaseModel

# Initialize the API router for inference endpoints
inference_router = APIRouter()

# Load the model and tokenizers
loader = LoadModel()
ar_tokenizer = loader.ar_tokenizer
en_tokenizer = loader.en_tokenizer
model = loader.model

# Define the response model for the output
class OutputResponse(BaseModel):
    translation: str

# Initialize the inference controller
inference_controller = InferenceController()

# Define the inference endpoint
@inference_router.get('/inference')
async def inference(sentence: str) -> OutputResponse:
    try:
        # Perform the translation using the inference controller
        translation = inference_controller.translate(sentence, en_tokenizer, ar_tokenizer, model)
        return OutputResponse(translation=translation)
    
    except Exception as e:
        # Raise an HTTP exception if there is an error
        raise HTTPException(status_code=500, detail=str(e))
