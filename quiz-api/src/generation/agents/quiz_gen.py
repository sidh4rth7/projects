import asyncio
from generation.prompts.quiz_gen import (
    ShortTextPrompt, MultipleChoiceSinglePrompt, MultipleChoiceMultiplePrompt, TrueFalsePrompt, CodingPrompt
)
from generation.schema.question_types import (
    ShortText, MultipleChoiceSingleAnswer, MultipleChoiceMultipleAnswer, TrueFalse, Coding
)
from utils.complexity_distribute import complexity_distribution
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json

load_dotenv()

async def create_question_chain(user_requirement: str, num_questions: int, complexity: list[str], 
                               PromptTemplate, schema) -> dict:
    try:
        prompt = PromptTemplate.partial(complexity=complexity, num_questions=num_questions)
        question_chain = prompt | ChatOpenAI(model="gpt-4o").with_structured_output(schema)
        result = await question_chain.ainvoke(user_requirement)
        return result.model_dump()
    except Exception as e:
        print(f"Error generating {schema.__name__} questions: {e}")
        return {}

async def generate_quiz(input: dict) -> dict:
    required_keys = {"user_requirement", "complexity", "question"}
    if not required_keys.issubset(input.keys()):
        raise ValueError(f"Input data is missing required keys: {required_keys - input.keys()}")

    distribution_result = complexity_distribution(input["question"], input["complexity"])

    question_mapping = {
        "ShortText": (ShortTextPrompt, ShortText),
        "MultipleChoiceSingleAnswer": (MultipleChoiceSinglePrompt, MultipleChoiceSingleAnswer),
        "MultipleChoiceMultipleAnswer": (MultipleChoiceMultiplePrompt, MultipleChoiceMultipleAnswer),
        "TrueFalse": (TrueFalsePrompt, TrueFalse),
        "Coding": (CodingPrompt, Coding)
    }

    tasks = []
    for question_type, (prompt, schema) in question_mapping.items():
        total_questions = input["question"].get(question_type, 0)
        complexity = distribution_result.get(question_type, [])
        if total_questions > 0:
            tasks.append(
                asyncio.create_task(
                    create_question_chain(input["user_requirement"], total_questions, complexity, prompt, schema)
                )
            )

    results = await asyncio.gather(*tasks)

    quiz_output = [
        {"type": q_type, "questions": result}
        for q_type, result in zip(question_mapping.keys(), results) if result
    ]

    return json.dumps(quiz_output, indent=4)