from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from app.agents.schemas import InsuranceAgentState
from app.core.logging import get_logger

logger = get_logger(__name__)

# Keep a reserve for system prompts, tool schemas, and messages produced during
# an agent's tool loop. Tune these against the configured model's context size.
PARENT_CONTEXT_TRIGGER_TOKENS = 400_000
PARENT_CONTEXT_KEEP_TOKENS = 40_000
AGENT_CONTEXT_TRIGGER_TOKENS = 450_000
AGENT_CONTEXT_KEEP_TOKENS = 50_000
SUMMARY_INPUT_MAX_TOKENS = 20_000

SUMMARY_PROMPT = """
你负责压缩客服对话历史，不回答用户问题。
只保留后续保险咨询需要的信息：用户目标、已确认事实、约束条件、涉及的产品和保障信息、已经给出的结论、尚未解决的问题。
不要补充推测或改变原意。使用简洁中文摘要。
单次摘要最多 500 个汉字。
""".strip()


def create_agent_summarization_middleware(model) -> SummarizationMiddleware:
    """在 agent 的每次模型调用前，压缩超出预算的消息历史。"""

    return SummarizationMiddleware(
        model=model,
        trigger=("tokens", AGENT_CONTEXT_TRIGGER_TOKENS),
        keep=("tokens", AGENT_CONTEXT_KEEP_TOKENS),
    )


def create_parent_context_compression_node(model):
    """为父图创建一次性的会话历史压缩节点。"""

    async def compress_context(state: InsuranceAgentState) -> dict:
        messages = state.get("messages", [])
        compacted_messages = await compact_messages(model, messages)
        if compacted_messages is None:
            return {}

        # MessagesState uses an append/merge reducer, so explicitly clear the
        # prior list before writing the summary and the retained recent turns.
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *compacted_messages,
            ]
        }

    return compress_context


async def compact_messages(
    model,
    messages: list[BaseMessage],
    *,
    trigger_tokens: int = PARENT_CONTEXT_TRIGGER_TOKENS,
    keep_tokens: int = PARENT_CONTEXT_KEEP_TOKENS,
) -> list[BaseMessage] | None:
    """Summarize old messages and return a compact prompt history if needed."""

    token_count = count_tokens_approximately(messages)
    if token_count <= trigger_tokens:
        return None

    split_at = _recent_turn_start(messages, keep_tokens)
    if split_at <= 0:
        # Keep the current user message intact; there is no older history to
        # summarize if it alone exceeds the budget.
        logger.warning(
            "会话上下文超过预算，但没有可压缩的旧消息",
            token_count=token_count,
        )
        return None

    old_messages = messages[:split_at]
    recent_messages = messages[split_at:]
    summary_text = await _summarize_history(model, old_messages)
    if not summary_text:
        return None

    logger.info(
        "会话上下文已压缩",
        token_count=token_count,
        summarized_messages=len(old_messages),
        retained_messages=len(recent_messages),
    )
    return [SystemMessage(content=f"此前对话摘要：\n{summary_text}"), *recent_messages]


async def _summarize_history(model, messages: list[BaseMessage]) -> str:
    """Summarize long histories in bounded, turn-aligned chunks."""

    summaries = []
    for chunk in _history_chunks(messages):
        summary = await model.ainvoke([SystemMessage(content=SUMMARY_PROMPT), *chunk])
        if summary.text.strip():
            summaries.append(summary.text.strip())

    while len(summaries) > 1:
        groups = _summary_groups(summaries)
        summaries = []
        for group in groups:
            summary = await model.ainvoke(
                [
                    SystemMessage(content=SUMMARY_PROMPT),
                    HumanMessage(
                        content="请合并并压缩以下分段摘要，保留后续咨询所需的关键信息：\n\n"
                        + "\n\n".join(group)
                    ),
                ]
            )
            if summary.text.strip():
                summaries.append(summary.text.strip())

    return summaries[0] if summaries else ""


def _history_chunks(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    chunks = []
    start = 0
    while start < len(messages):
        end = start
        chunk_tokens = 0
        while end < len(messages):
            message_tokens = count_tokens_approximately([messages[end]])
            if end > start and chunk_tokens + message_tokens > SUMMARY_INPUT_MAX_TOKENS:
                break
            chunk_tokens += message_tokens
            end += 1

        # Keep each user turn and its assistant/tool exchange together.
        while end < len(messages) and not isinstance(messages[end], HumanMessage):
            end += 1

        chunks.append(messages[start:end])
        start = end
    return chunks


def _summary_groups(summaries: list[str]) -> list[list[str]]:
    groups = []
    current = []
    current_tokens = 0
    for summary in summaries:
        summary_tokens = count_tokens_approximately([HumanMessage(content=summary)])
        if current and current_tokens + summary_tokens > SUMMARY_INPUT_MAX_TOKENS:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(summary)
        current_tokens += summary_tokens
    if current:
        groups.append(current)
    return groups


def _recent_turn_start(messages, keep_tokens: int = PARENT_CONTEXT_KEEP_TOKENS) -> int:
    """Return a suffix boundary that retains complete recent user turns."""

    if not messages:
        return 0

    start = len(messages)
    retained_tokens = 0
    while start > 0:
        message_tokens = count_tokens_approximately([messages[start - 1]])
        if start < len(messages) and retained_tokens + message_tokens > keep_tokens:
            break
        retained_tokens += message_tokens
        start -= 1

    # Avoid splitting an assistant/tool exchange or starting with an orphaned
    # tool result. Keep the entire turn beginning at its user message.
    while start > 0 and not isinstance(messages[start], HumanMessage):
        start -= 1

    return start
