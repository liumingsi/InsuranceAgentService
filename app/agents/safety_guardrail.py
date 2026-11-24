from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langgraph.graph.state import CompiledStateGraph

from app.agents.schemas import GuardrailResult
from app.core.config import settings
from app.core.logging import get_logger
from .context_compression import create_agent_summarization_middleware

logger = get_logger(__name__)

SYSTEM_PROMPT = """
你是一名数据安全专家，防止sql注入和xx攻击，确保数据安全。
对用户输入的信息进行判断，如果存在sql注入或xx攻击，则返回错误信息，否则返回用户输入的信息。
"""


# agent在整个系统中，只初始化一次，
async def init_safety_guardrail() -> CompiledStateGraph:
    # 1.初始化模型
    model = init_chat_model(
        model=settings.llm.chat_model,
        api_key=settings.llm.api_key,
        extra_body={"thinking": {"type": "disabled"}}
    )

    # 2.创建Agent，绑定model、tools、Checkpointer
    agent = create_agent(
        model=model,
        middleware=[
            create_agent_summarization_middleware(model),
            # 用户输入中的邮箱，直接打码替换
            PIIMiddleware("email", strategy="redact", apply_to_input=True),

            # 信用卡号，部分遮盖保留可读性
            PIIMiddleware("credit_card", strategy="mask"),

            # 工具返回结果中如果出现 IP，进行脱敏
            PIIMiddleware("ip", strategy="redact", apply_to_tool_results=True),

            # 自定义规则：检测到 API Key 直接阻断
            PIIMiddleware("api_key", detector=r"sk-[a-zA-Z0-9]{32}", strategy="block"),
        ],
        system_prompt=SYSTEM_PROMPT,
        response_format=ToolStrategy(schema=GuardrailResult),
    )
    return agent
