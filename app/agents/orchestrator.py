import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.constants import START, END
from langgraph.errors import NodeError
from langgraph.graph import StateGraph
from langgraph.types import Command, interrupt, TimeoutPolicy, RetryPolicy
from langchain.chat_models import init_chat_model

from .claim import init_claim_advisor
from .schemas import InsuranceAgentState, Intent
from .intent_router import router
from .chitchat import init_chitchat_agent
from .insurance_advisor import init_insurance_advisor
from .safety_guardrail import init_safety_guardrail
from app.core.config import settings
# 1.定义节点

# 1.1.意图识别节点
async def route_node(state: InsuranceAgentState) -> Command[Intent]:
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
        goto=(
            "requirement_confirmation"
            if result.intent in {"recommendation_plan", "claim"}
            else result.intent
        )
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


def _confirmation_context(messages: list[BaseMessage]) -> str:
    """Use user statements and the parent-graph summary as confirmed context."""

    context_messages = [
        message.text
        for message in messages
        if isinstance(message, HumanMessage)
        or (
            isinstance(message, SystemMessage)
            and message.text.startswith("此前对话摘要：")
        )
    ]
    return "\n".join(context_messages)


def _recommendation_confirmation(context: str, latest_request: str) -> dict | None:
    request_terms = ("推荐", "适合我", "帮我选", "给我选", "配置保险", "买保险", "做方案", "定制方案")
    if not any(term in latest_request for term in request_terms):
        return None

    missing = []
    age_present = bool(re.search(
        r"\d{1,2}\s*岁|年龄\s*[:：]?\s*\d{1,2}|(?:19|20)\d{2}\s*年生|儿童|孩子|老人|婴儿|成年",
        context,
    ))
    sex_present = any(term in context for term in ("男性", "女性", "男", "女", "先生", "女士"))
    if not age_present or not sex_present:
        missing.append("被保险人的年龄和性别")

    coverage_types = ("医疗", "重疾", "意外", "寿险", "养老", "教育", "车险", "旅行")
    if not any(term in context for term in coverage_types):
        missing.append("最关注的保障类型")

    if not re.search(
        r"预算|保费|年缴|月缴|每年|每月|不限预算|不设预算|\d[\d,]*(?:\.\d+)?\s*(?:元|万|块)",
        context,
    ):
        missing.append("年预算范围")

    if not missing:
        return None

    question = "为了推荐更合适的保险方案，请补充：" + "、".join(missing) + "。"
    options = (
        ["医疗保障", "重疾保障", "意外保障", "寿险", "养老/教育", "还不确定"]
        if "最关注的保障类型" in missing
        else []
    )
    return {
        "type": "requirement_confirmation",
        "title": "确认保险需求",
        "question": question,
        "assumptions": [],
        "options": options,
    }


def _claim_confirmation(context: str, latest_request: str) -> dict | None:
    if any(term in latest_request for term in ("理赔流程", "理赔材料有哪些", "一般怎么理赔")):
        return None

    personal_claim_terms = (
        "能赔",
        "赔多少",
        "能不能赔",
        "能否理赔",
        "报销多少",
        "我这次",
        "我的保单",
        "我住院",
        "我出险",
        "申请理赔",
        "我要理赔",
    )
    if not any(term in latest_request for term in personal_claim_terms):
        return None

    missing = []
    policy_terms = ("保单", "保险单", "保险名称", "产品", "险种", "医疗险", "重疾险", "意外险", "寿险")
    if not any(term in context for term in policy_terms):
        missing.append("涉及的保单或保险产品")

    incident_terms = (
        "住院", "门诊", "疾病", "诊断", "手术", "事故", "出险", "受伤", "车祸", "治疗", "报案", "日期", "时间"
    )
    if not any(term in context for term in incident_terms):
        missing.append("出险或就诊经过及时间")

    if not missing:
        return None

    return {
        "type": "requirement_confirmation",
        "title": "确认理赔信息",
        "question": "为了继续判断，请补充：" + "、".join(missing) + "。",
        "assumptions": [],
        "options": [],
    }


async def requirement_confirmation_node(state: InsuranceAgentState):
    """Pause before claim/recommendation agents when material details are missing."""

    workflow = state.get("active_workflow")
    messages = state.get("messages", [])
    context = _confirmation_context(messages)
    latest_request = next(
        (message.text for message in reversed(messages) if isinstance(message, HumanMessage)),
        "",
    )
    if workflow == "recommendation_plan":
        confirmation = _recommendation_confirmation(context, latest_request)
    elif workflow == "claim":
        confirmation = _claim_confirmation(context, latest_request)
    else:
        confirmation = None

    if confirmation is None:
        return Command(goto=workflow)

    answer = interrupt(confirmation)
    if isinstance(answer, dict):
        answer_text = str(answer.get("answer", ""))
    else:
        answer_text = str(answer)
    return Command(
        update={
            "messages": [
                AIMessage(content=confirmation["question"]),
                HumanMessage(content=answer_text),
            ]
        },
        goto=workflow,
    )

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

async def human_handoff_node(state: InsuranceAgentState):
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
async def fallback_node(state: InsuranceAgentState):
    return {"messages": [AIMessage("无法识别您的意图，请再说一遍~~")]}
# 节点错误处理
async def default_error_handler(error: NodeError):
    return {"status": f"handled: {error.error}"}
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
    # 下面的节点都会采用这个的重试和超时策略
    builder.set_node_defaults(
        retry_policy=RetryPolicy(max_attempts=3),  # 重试策略
        error_handler=default_error_handler,
        timeout=TimeoutPolicy(run_timeout=30), # 超时策略
    )
    # 添加节点
    builder.add_node("route", route_node)
    builder.add_node("requirement_confirmation", requirement_confirmation_node)
    builder.add_node("recommendation_plan",await init_insurance_advisor())
    builder.add_node("chitchat", await init_chitchat_agent())
    builder.add_node("claim", await init_claim_advisor())
    builder.add_node("human_handoff", human_handoff_node)
    builder.add_node("fallback", fallback_node)
    builder.add_node("safety_guardrail", await init_safety_guardrail())

    # 添加Edge，连Node
    builder.add_edge(START, "safety_guardrail")
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
