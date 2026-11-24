from langchain_core.messages import AIMessage, BaseMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import Command
from langchain.chat_models import init_chat_model

from .claim import init_claim_advisor
from .schemas import InsuranceAgentState, Intent
from .intent_router import router
from .chitchat import init_chitchat_agent
from .context_compression import create_parent_context_compression_node
from .insurance_advisor import init_insurance_advisor
from .safety_guardrail import init_safety_guardrail
from app.core.config import settings
# 1.定义节点

# 1.1.意图识别节点
def route_node(state: InsuranceAgentState) -> Command[Intent]:
    # 获取上一轮的路由结果
    previous_workflow = state.get("active_workflow", "")
    # 获取用户本轮的提问问题
    messages = state['messages']
    # 准备上下文
    previous_ai_message = find_previous_ai_message(messages)
    context = f"上一轮工作流：{previous_workflow} \n 上一轮AI回复的信息：{previous_ai_message}"

    # 调用路由工具，完成意图识别
    result = router.route(messages[-1].text, context)

    if isinstance(result, str):
        # 当前识别出的是寒暄，直接返回固定回复结果
        return Command(
            update={
                "active_workflow": "chitchat",
                "previous_workflow": previous_workflow,
                "messages": [AIMessage(content=result)]
            },
            goto=END
        )

    return Command(
        update={
            "active_workflow": result.intent,
            "previous_workflow": previous_workflow
        },
        goto=result.intent
    )

def find_previous_ai_message(messages: list[BaseMessage]) -> str:
    """查找本轮用户消息之前最近一条可展示的AI回复"""

    for message in reversed(messages[:-1]):
        # 跳过非AIMessage
        if not isinstance(message, AIMessage):
            continue
        # 跳过工具消息
        if message.tool_calls:
            continue
        # 如果是正常AIMessage，直接返回
        if message.text:
            return message.text
    # 如果是空消息
    return ""

# 1.2.保险方案推荐咨询节点
# def recommendation_node(state: InsuranceAgentState):
#     return {"messages": [AIMessage("保险方案推荐功能尚未完成~")]}

# # 1.3.理赔咨询节点
# def claim_node(state: InsuranceAgentState):
#     return {"messages": [AIMessage("理赔咨询功能尚未完成~")]}

# 1.4.闲聊节点
# def chitchat_node(state: InsuranceAgentState):
#     return {"messages": [AIMessage("哈哈，欢迎你来到安心保商城，今天聊点啥~")]}

# 1.5.人工
from langchain_core.messages import AIMessage

def human_handoff_node(state: InsuranceAgentState):
    image_url = "https://miss-9527.oss-cn-beijing.aliyuncs.com/1.jpg"
    return {
        "messages": [
            AIMessage(
                content=[
                    {
                        "type": "text",
                        "text": "请添加人工客户微信，下面是人工客服微信。",
                    },
                    {
                        "type": "image",
                        "image_url": {"url": image_url},
                    },
                ]
            )
        ]
    }

# 1.6.fallback
def fallback_node(state: InsuranceAgentState):
    return {"messages": [AIMessage("无法识别您的意图，请再说一遍~~")]}
# 判断数据是否安全
def route_after_guardrail(state: InsuranceAgentState) -> str:
    sr = state.get("structured_response")
    if sr is None:
        return "blocked"                # fail-closed，宁可拦错也不放行

    # 兼容 Pydantic 对象和 dict 两种形态
    if isinstance(sr, dict):
        safe = bool(sr.get("safe", False))
    else:
        safe = bool(getattr(sr, "safe", False))

    return "pass" if safe else "blocked"
# 2.创建Graph，用Edge连接Node
async def init_insurance_agent(checkpointer):
    # 建造器
    builder = StateGraph(InsuranceAgentState)

    context_model = init_chat_model(
        model=settings.llm.chat_model,
        api_key=settings.llm.api_key,
        extra_body={"thinking": {"type": "disabled"}},
    )

    # 添加节点
    builder.add_node(
        "context_compression",
        create_parent_context_compression_node(context_model),
    )
    builder.add_node("route", route_node)
    builder.add_node("recommendation_plan",await init_insurance_advisor())
    builder.add_node("chitchat", init_chitchat_agent())
    builder.add_node("claim", await init_claim_advisor())
    builder.add_node("human_handoff", human_handoff_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("safety_guardrail", await init_safety_guardrail())

    # 添加Edge，连Node
    builder.add_edge(START, "context_compression")
    builder.add_edge("context_compression", "safety_guardrail")
    builder.add_conditional_edges(  # 添加条件边
        "safety_guardrail",
        route_after_guardrail,
        {
            "pass": "route",
            "blocked": "fallback",
        },
    )
    builder.add_edge("recommendation_plan", END)
    builder.add_edge("claim", END)
    builder.add_edge("human_handoff", END)
    builder.add_edge("fallback", END)
    builder.add_edge("chitchat", END)

    # 编译Graph
    return builder.compile(checkpointer=checkpointer)
