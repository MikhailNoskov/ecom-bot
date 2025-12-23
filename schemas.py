from typing import List, Dict
from pydantic import BaseModel


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