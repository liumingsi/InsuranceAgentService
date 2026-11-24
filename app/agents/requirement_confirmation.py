from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.types import interrupt


COMMON_CONFIRMATION_PROMPT = """
## 需求澄清与用户确认
- 如果用户的目标存在多种合理解释，或缺少会实质影响结论、方案或后续处理的关键信息，先调用 `ask_user_for_clarification`，等待用户确认后再继续。
- 调用该工具时，只提出继续处理所必需的简短、具体问题；如果有需要确认的假设或选项，一并提供，方便前端展示。
- 在用户回答前，不要调用其他业务查询工具、不要自行补全事实，也不要给出依赖这些事实的最终判断和建议。
- 对一般性知识问题，如果缺失信息不会改变答案，直接回答。用户补充信息后结合对话历史继续处理，不要重复询问已确认的内容。
""".strip()


@tool
def ask_user_for_clarification(
    question: str,
    assumptions: list[str] | None = None,
    options: list[str] | None = None,
) -> str:
    """Pause for the user's clarification before continuing an uncertain request."""

    response = interrupt(
        {
            "type": "requirement_confirmation",
            "title": "请确认需求",
            "question": question,
            "assumptions": assumptions or [],
            "options": options or [],
        }
    )
    if isinstance(response, dict):
        return str(response.get("answer", ""))
    return str(response)


class RequirementConfirmationMiddleware(AgentMiddleware):
    """Adds an interrupting clarification tool and its usage policy to an agent."""

    def __init__(self, domain_prompt: str) -> None:
        super().__init__()
        self.domain_prompt = domain_prompt
        self.tools = [ask_user_for_clarification]

    def _with_confirmation_prompt(self, request: ModelRequest) -> ModelRequest:
        base_prompt = request.system_message.text if request.system_message else ""
        system_prompt = "\n\n".join(
            part
            for part in (base_prompt, COMMON_CONFIRMATION_PROMPT, self.domain_prompt)
            if part
        )
        return request.override(system_message=SystemMessage(content=system_prompt))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        return handler(self._with_confirmation_prompt(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        return await handler(self._with_confirmation_prompt(request))

    def _allow_confirmation_tool_only(self, state) -> dict[str, Any] | None:
        messages = state.get("messages", [])
        last_ai_message = next(
            (message for message in reversed(messages) if isinstance(message, AIMessage)),
            None,
        )
        if last_ai_message is None or not last_ai_message.tool_calls:
            return None

        confirmation_calls = [
            call
            for call in last_ai_message.tool_calls
            if call.get("name") == "ask_user_for_clarification"
        ]
        if not confirmation_calls:
            return None

        # Never let business tools run in parallel with a clarification pause.
        filtered_message = last_ai_message.model_copy(
            update={"tool_calls": confirmation_calls[:1]}
        )
        if len(confirmation_calls) == len(last_ai_message.tool_calls) == 1:
            return None
        return {"messages": [filtered_message]}

    def after_model(self, state, runtime) -> dict[str, Any] | None:
        return self._allow_confirmation_tool_only(state)

    async def aafter_model(self, state, runtime) -> dict[str, Any] | None:
        return self._allow_confirmation_tool_only(state)


def create_requirement_confirmation_middleware(
    domain_prompt: str,
) -> RequirementConfirmationMiddleware:
    return RequirementConfirmationMiddleware(domain_prompt)
