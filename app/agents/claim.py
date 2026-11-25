from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from langchain.chat_models import init_chat_model
from langgraph.graph.state import CompiledStateGraph

from app.core.config import settings
from app.core.logging import get_logger
from .context_compression import create_agent_summarization_middleware
from .requirement_confirmation import create_requirement_confirmation_middleware
from .tools import query_candidate_products, retrieve_insurance_clause

logger = get_logger(__name__)

SYSTEM_PROMPT = """
你是“安心保”的智能保险顾问。
你需要使用专业、准确、容易理解的语言回答用户的理赔问题。
当信息不足时，应先向用户追问，不要编造信息。
当用户询问理赔条款问题时，必须调用工具查询条款，然后根据条款回答，不能自己编造。

当调用 retrieve_insurance_clause 后：
- 对条款相关结论，在正文用 `[S1]` 这样的 citation_id 标记依据。
- 只使用检索结果中的 citation_id，不要在正文粘贴长篇条款；前端会单独展示“引用来源”卡片及短原文摘录。
- 针对同一问题最多调用一次检索工具，最多使用最相关的4条来源。
- 如果检索结果没有能支持结论的原文，明确说明无法从当前条款片段确认，不要给出确定的责任结论。
"""


# agent在整个系统中，只初始化一次，
async def init_claim_advisor() -> CompiledStateGraph:
    # 1.初始化模型
    model = init_chat_model(
        model=settings.llm.chat_model,
        api_key=settings.llm.api_key,
        extra_body={"thinking": {"type": "disabled"}}
    )

    # 2.创建Agent，绑定model、tools、Checkpointer
    agent = create_agent(
        model=model,
        tools=[query_candidate_products, retrieve_insurance_clause],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            TodoListMiddleware(),
            create_requirement_confirmation_middleware(
                "理赔咨询中，若出险时间、事故或治疗经过、涉及的保单/产品、当前理赔阶段等信息会影响责任判断或办理建议，先确认必要信息。用户只询问通用流程时可直接概述，并说明具体结论仍需结合保单条款。"
            ),
            create_agent_summarization_middleware(model),
        ],
    )

    logger.info("保险理赔Agent初始化完毕~✅️")
    return agent
