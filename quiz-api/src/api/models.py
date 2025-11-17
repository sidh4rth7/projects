from pydantic import BaseModel
from typing import List, Optional

class QuestionRequest(BaseModel):
    user_requirement: str
    complexity: List[str]
    question: dict

class QuestionResponse(BaseModel):
    type: str
    questions: List[dict]

class QuizResponse(BaseModel):
    quiz: List[QuestionResponse]