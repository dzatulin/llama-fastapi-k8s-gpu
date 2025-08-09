from pydantic import BaseModel
from typing import Optional

class ChatMessage(BaseModel):
    turn: str
    message: str

class BotProfile(BaseModel):
    name: str
    appearance: str
    system_prompt: Optional[str] = ""

class UserProfile(BaseModel):
    name: str

class BotMessageRequest(BaseModel):
    bot_profile: BotProfile
    user_profile: UserProfile
    context: list[ChatMessage]
