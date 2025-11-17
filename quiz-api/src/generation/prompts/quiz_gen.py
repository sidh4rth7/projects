from langchain.prompts import PromptTemplate

class ShortTextPrompt(PromptTemplate):
    template: str = "Generate a short answer question based on the following requirement: {requirement}"

class MultipleChoiceSinglePrompt(PromptTemplate):
    template: str = "Create a multiple choice question with one correct answer based on: {requirement}"

class MultipleChoiceMultiplePrompt(PromptTemplate):
    template: str = "Generate a multiple choice question with multiple correct answers based on: {requirement}"

class TrueFalsePrompt(PromptTemplate):
    template: str = "Formulate a true/false question based on the following requirement: {requirement}"

class CodingPrompt(PromptTemplate):
    template: str = "Create a coding question based on the following requirement: {requirement}"