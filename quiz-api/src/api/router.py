from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from generation.agents.quiz_gen import generate_quiz

router = APIRouter()

class QuizRequest(BaseModel):
    user_requirement: str
    complexity: list[str]
    question: dict

class QuizResponse(BaseModel):
    type: str
    questions: list

@router.post("/generate-quiz", response_model=list[QuizResponse])
async def create_quiz(quiz_request: QuizRequest):
    try:
        result = await generate_quiz(quiz_request.dict())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))