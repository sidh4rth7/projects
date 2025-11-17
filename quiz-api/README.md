# README.md

# Quiz API

This project is a FastAPI application designed for generating quiz questions based on user requirements. It utilizes asynchronous programming to efficiently handle requests and generate various types of quiz questions.

## Project Structure

```
quiz-api
├── src
│   ├── api
│   │   ├── __init__.py
│   │   ├── router.py
│   │   └── models.py
│   ├── generation
│   │   ├── __init__.py
│   │   ├── agents
│   │   │   ├── __init__.py
│   │   │   └── quiz_gen.py
│   │   ├── prompts
│   │   │   ├── __init__.py
│   │   │   └── quiz_gen.py
│   │   └── schema
│   │       ├── __init__.py
│   │       └── question_types.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── complexity_distribute.py
│   ├── config.py
│   └── main.py
├── tests
│   ├── __init__.py
│   └── test_api.py
├── .env
├── requirements.txt
└── README.md
```

## Features

- Asynchronous quiz question generation
- Support for multiple question types including short text, multiple choice, true/false, and coding questions
- Complexity distribution for questions based on user-defined parameters

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd quiz-api
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Start the FastAPI application:
   ```
   uvicorn src.main:app --reload
   ```

2. Access the API documentation at `http://127.0.0.1:8000/docs`.

## Testing

To run the tests, use the following command:
```
pytest
```

## Environment Variables

Make sure to create a `.env` file in the root directory with the necessary environment variables for your application.

## License

This project is licensed under the MIT License.