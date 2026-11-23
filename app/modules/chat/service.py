from fastapi.sse import ServerSentEvent
from langchain.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.chat_thread.repository import ChatThreadRepository
from app.modules.chat_thread.exceptions import ChatThreadNotFound
from .schema import ChatRequest


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

        # 2.组织消息
        _input = {"messages": [HumanMessage(request.message)]}

        # 3.组织config，实现会话记忆
        config = {"configurable": {"thread_id": request.thread_id}}

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

        async for event in stream:
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

        yield ServerSentEvent(data="END", event="done")