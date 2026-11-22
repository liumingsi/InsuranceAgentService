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
            method = event['method']
            if method == 'messages':
                data = event['params']['data'][0]

                # 统一提取文本和图片块（兼容 AIMessage 和 AIMessageChunk）
                text_parts = []
                image_urls = []

                # 情况 1：content 是列表（多模态消息）
                if isinstance(data.content, list):
                    for block in data.content:
                        block_type = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)

                        if block_type == "text":
                            text_parts.append(block.get("text") if isinstance(block, dict) else block.text)

                        elif block_type == "image":
                            # OpenAI 兼容格式
                            image_url = block.get("image_url", {}).get("url") if isinstance(block, dict) else None
                            if image_url:
                                image_urls.append(image_url)

                # 情况 2：content 是纯字符串（无多模态的简单场景）
                elif isinstance(data.content, str):
                    text_parts.append(data.content)

                # 兼容流式 chunk 的 delta 格式
                elif isinstance(data, dict) and data.get('delta') and data['delta'].get('text'):
                    text_parts.append(data['delta']['text'])

                # 组装输出
                text = "".join(text_parts)

                if text:
                    yield ServerSentEvent(data=text, event="message")

                # 图片 URL 单独作为事件发出，避免混在文本流里
                for url in image_urls:
                    yield ServerSentEvent(data=url, event="image")

        yield ServerSentEvent(data="END", event="done")