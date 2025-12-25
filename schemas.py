from typing import List, Dict
from pydantic import BaseModel, Field


class ToneSchema(BaseModel):
    role: str
    persona: str
    sentences_max: int
    bullets: bool = True
    avoid: List[str]
    must_include: List[str]


class FieldsSchema(BaseModel):
    answer: str
    tone: str
    actions: str


class FormatSchema(BaseModel):
    fields: FieldsSchema


class StyleSchema(BaseModel):
    brand: str
    tone: ToneSchema
    task: str
    rules: List[str]
    fallback: Dict[str, str]
    format: FormatSchema


class UsageSchema(BaseModel):
    completion_tokens: int | None = 0
    prompt_tokens: int | None = 0
    total_tokens: int | None = 0


class ReplySchema(FieldsSchema):
    answer: str
    tone: str | None = None
    actions: List[str] = []
    usage: UsageSchema


class Grade(BaseModel):
    score: int = Field(..., ge=0, le=100)
    notes: str