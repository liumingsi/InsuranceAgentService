from uuid import UUID

from pydantic import Field, BaseModel

class ChatRequest(BaseModel):
    """基础对话请求"""

    thread_id: UUID
    message: str = Field(min_length=1)
    resume: bool = Field(
        default=False,
        description="当前消息是对上一条人工确认中断的回复时设为 true",
    )
