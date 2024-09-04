from fastapi import APIRouter

# Initialize the API router for base endpoints
base_router = APIRouter()

# Define the welcome endpoint
@base_router.get('/')
async def get_welcome_user() -> dict:
    # Return a welcome message
    return {
        'messages': 'hello, how are you'
    }
