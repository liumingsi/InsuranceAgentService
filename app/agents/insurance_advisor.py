from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware, SummarizationMiddleware, HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langgraph.graph.state import CompiledStateGraph

from app.core.config import settings
from app.core.logging import get_logger
from .tools import query_candidate_products, retrieve_insurance_clause, ask_user

logger = get_logger(__name__)

SYSTEM_PROMPT = """
你是“安心保”的智能保险顾问。
...
当信息不足，无法基于现有信息给出可靠回答时，**必须调用 `ask_user` 工具**，并在 `reason` 参数中简要说明需要用户补充哪些信息。
不要自行生成追问文本，也不要猜测用户的意图。必须等待 `ask_user` 工具返回人工补充的信息后，再继续推理。

当调用 retrieve_insurance_clause 后：
- 对条款相关结论，在正文用 `[S1]` 这样的 citation_id 标记依据。
- 只使用检索结果中的 citation_id，不要在正文粘贴长篇条款；前端会单独展示“引用来源”卡片及短原文摘录。
- 针对同一问题最多调用一次检索工具，最多使用最相关的4条来源。
- 如果检索结果没有能支持结论的原文，明确说明无法从当前条款片段确认，不要给出确定的保障结论。
"""


# agent在整个系统中，只初始化一次，
async def init_insurance_advisor() -> CompiledStateGraph:
    # 1.初始化模型
    model = init_chat_model(
        model=settings.llm.chat_model,
        api_key=settings.llm.api_key,
        extra_body={"thinking": {"type": "disabled"}}
    )

    # 2.创建Agent，绑定model、tools、Checkpointer
    agent = create_agent(
        model=model,
        tools=[query_candidate_products, retrieve_insurance_clause, ask_user],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            TodoListMiddleware(),
            SummarizationMiddleware(
                model=model,
                max_tokens_before_summary=40,  # 在 4000 个 token 时触发摘要
                messages_to_keep=2,  # 摘要后保留最近 20 条消息
                summary_prompt="Custom prompt for summarization...",  # 可选
            ),
            HumanInTheLoopMiddleware(
                interrupt_on={"ask_user": {"allowed_decisions": ["respond"]}},
            )
        ],
    )
    logger.info("保险顾问Agent初始化完毕~✅️")
    return agent
