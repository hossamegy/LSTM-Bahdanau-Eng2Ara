from fastapi import FastAPI
from routers import base_router, inference_router
from fastapi.middleware.cors import CORSMiddleware

# Create an instance of the FastAPI application
app = FastAPI()

# Add CORS middleware to handle Cross-Origin Resource Sharing
# This configuration allows requests from any origin (useful for development and testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from any origin
    allow_credentials=True,  # Allows credentials to be included in requests
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Include the base router for handling base endpoints
app.include_router(base_router)

# Include the inference router for handling inference-related endpoints
app.include_router(inference_router)
