from fastapi import FastAPI
from api.router import router

app = FastAPI()

app.include_router(router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Quiz API!"}