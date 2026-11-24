from typing import Literal

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

# 路由意图常量
Intent = Literal[
    "chitchat",
    "recommendation_plan",
    "claim",
    "human_handoff",
    "fallback",
]
class GuardrailResult(BaseModel):
    """安全校验的结构化输出"""
    safe: bool = Field(description="输入是否安全；无 SQL 注入 / XSS / PII 泄漏时为 true")
    reason: str = Field(default="", description="不安全时的原因说明")
    category: Literal["none", "sql_injection", "xss", "pii", "other"] = Field(
        default="none", description="风险类别"
    )
    sanitized_input: str = Field(default="", description="清洗/脱敏后的输入")

class RouteResult(BaseModel):
    """意图识别结果"""

    intent: Intent = Field(description="本轮消息应该进入的工作流")
    reason: str = Field(description="判断该意图的理由")


class InsuranceAgentState(MessagesState):
    """保险顾问主Graph状态"""

    previous_workflow: Intent
    active_workflow: Intent
    chitchat_count: int
    structured_response: GuardrailResult | None = None