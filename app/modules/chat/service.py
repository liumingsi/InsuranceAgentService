import json

from fastapi import HTTPException
from fastapi.sse import ServerSentEvent
from langchain.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.chat_thread.repository import ChatThreadRepository
from app.modules.chat_thread.exceptions import ChatThreadNotFound
from .schema import ChatRequest
from .sources import source_cards_for_latest_turn


def _interrupt_payload(pending_interrupt) -> tuple[str, dict]:
    if isinstance(pending_interrupt, dict):
        value = pending_interrupt.get("value", pending_interrupt)
        interrupt_id = pending_interrupt.get("id") or pending_interrupt.get("interrupt_id")
    else:
        value = getattr(pending_interrupt, "value", pending_interrupt)
        interrupt_id = getattr(pending_interrupt, "id", None)

    if isinstance(value, dict):
        payload = dict(value)
    else:
        payload = {"type": "requirement_confirmation", "question": str(value)}
    if interrupt_id is not None:
        payload["interrupt_id"] = interrupt_id

    dedupe_key = str(interrupt_id) if interrupt_id is not None else json.dumps(payload, ensure_ascii=False)
    return dedupe_key, payload


class ChatService:

    def __init__(self, session: AsyncSession, agent: CompiledStateGraph):
        self.agent = agent
        self.repository = ChatThreadRepository(session)

    async def chat_stream(self, user_id: int, request: ChatRequest):
        """实现Agent流式对话的函数"""

        # 1.校验，判断当前thread_id是否属于当前用户
        chat_thread = await self.repository.required_owner(user_id, request.thread_id)
        if not chat_thread:
            # 指定会话不属于当前用户
            raise ChatThreadNotFound()

        # 2.组织config，实现会话记忆
        config = {"configurable": {"thread_id": request.thread_id}}

        # 3.普通消息启动新一轮；人工确认回复则恢复暂停的图执行。
        if request.resume:
            snapshot = await self.agent.aget_state(config)
            if not snapshot.interrupts:
                raise HTTPException(status_code=409, detail="当前会话没有等待确认的中断")
            _input = Command(resume=request.message)
        else:
            _input = {"messages": [HumanMessage(request.message)]}

        # 4.调用Agent
        stream = await self.agent.astream_events(
            _input,
            config,
            version="v3"
        )

        # 5.返回结果
        # async for event in stream:
        #     method = event['method']
        #     # 判断事件类型，可以是messages、interrupts等等
        #     if method == 'messages':
        #         data = event['params']['data'][0]
        #         if isinstance(data, AIMessage):
        #             yield ServerSentEvent(data=data.text, event="message")
        #         elif data.get('delta') and data['delta'].get('text'):
        #             yield ServerSentEvent(data=data['delta'].get('text'), event="message")
        #
        # yield ServerSentEvent(data="END", event="done")

        seen_interrupts = set()
        latest_messages = []
        async for event in stream:
            params = event.get("params", {})

            # v3 values events contain graph state, including ToolMessages
            # returned by retrieve_insurance_clause.
            if event.get("method") == "values":
                state = params.get("data")
                if isinstance(state, dict) and isinstance(state.get("messages"), list):
                    latest_messages = state["messages"]

            for pending_interrupt in params.get("interrupts", ()):
                dedupe_key, payload = _interrupt_payload(pending_interrupt)
                if dedupe_key in seen_interrupts:
                    continue
                seen_interrupts.add(dedupe_key)
                yield ServerSentEvent(
                    data=json.dumps(payload, ensure_ascii=False),
                    event="interrupt",
                )

            if event.get('method') != 'messages':
                continue

            data = event['params']['data'][0]

            text_parts = []
            image_urls = []

            # 先按类型提取，不提前访问 .content
            if isinstance(data, dict):
                content = data.get("content")
                delta = data.get("delta") or {}
                delta_text = delta.get("text") if isinstance(delta, dict) else None
            else:
                content = getattr(data, "content", None)
                delta_text = None

            # 情况 1：content 是列表（多模态）
            if isinstance(content, list):
                for block in content:
                    block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)

                    if block_type == "text":
                        text_parts.append(block.get("text") if isinstance(block, dict) else block.text)

                    elif block_type == "image":
                        image_url = (
                            block.get("image_url", {}).get("url")
                            if isinstance(block, dict)
                            else None
                        )
                        if image_url:
                            image_urls.append(image_url)

            # 情况 2：content 是字符串
            elif isinstance(content, str):
                text_parts.append(content)

            # 情况 3：兼容流式 delta
            elif delta_text:
                text_parts.append(delta_text)

            text = "".join(text_parts)
            if text:
                yield ServerSentEvent(data=text, event="message")

            for url in image_urls:
                yield ServerSentEvent(data=url, event="image")

        # LangGraph v3 exposes pending interrupts through the run's interrupt
        # projection. This also catches root-level interrupts not seen above.
        for pending_interrupt in await stream.interrupts():
            dedupe_key, payload = _interrupt_payload(pending_interrupt)
            if dedupe_key in seen_interrupts:
                continue
            seen_interrupts.add(dedupe_key)
            yield ServerSentEvent(
                data=json.dumps(payload, ensure_ascii=False),
                event="interrupt",
            )

        final_state = await stream.output()
        if isinstance(final_state, dict) and isinstance(final_state.get("messages"), list):
            latest_messages = final_state["messages"]

        source_cards = source_cards_for_latest_turn(latest_messages)
        if source_cards:
            yield ServerSentEvent(
                data=json.dumps(
                    {
                        "title": "引用来源",
                        "count": len(source_cards),
                        "sources": source_cards,
                    },
                    ensure_ascii=False,
                ),
                event="sources",
            )

        yield ServerSentEvent(data="END", event="done")
