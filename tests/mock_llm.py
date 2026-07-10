import re
from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult, ChatGeneration, LLMResult, Generation
from langchain_core.runnables import RunnableLambda
from pydantic import Field

class MockChatOpenAI(BaseChatModel):
    model_name: str = Field(default="gpt-4o-mini", alias="model")

    def _response_for_messages(self, messages: List[BaseMessage]) -> AIMessage:
        text = "\n".join(str(getattr(message, "content", "")) for message in messages)

        if "JSON" in text or "json" in text:
            match = re.search(r"提供(.+?)的天气信息", text)
            city = match.group(1) if match else "北京"
            return AIMessage(content=f'{{"city": "{city}", "temperature": "25℃", "condition": "晴"}}')

        if "逗号" in text or "分隔" in text:
            return AIMessage(content="提示词,模型,工具,链,记忆")

        return AIMessage(content="This is a mock response.")
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        message = self._response_for_messages(messages)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def bind_tools(self, tools: List[Any], **kwargs: Any) -> RunnableLambda:
        def invoke_with_tool_call(messages: Any) -> AIMessage:
            if any(getattr(message, "type", None) == "tool" for message in messages):
                return AIMessage(content="工具调用完成。")

            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tools[0].name,
                        "args": {"city": "北京"},
                        "id": "mock-tool-call-1",
                    }
                ],
            )

        return RunnableLambda(invoke_with_tool_call)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> RunnableLambda:
        def invoke_structured(messages: Any) -> Any:
            if hasattr(schema, "model_validate"):
                return schema.model_validate({"city": "北京"})
            return {"city": "北京"}

        return RunnableLambda(invoke_structured)

    @property
    def _llm_type(self) -> str:
        return "mock-chat-openai"

class MockOpenAI(BaseLLM):
    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = [[Generation(text="This is a mock LLM response.")]]
        return LLMResult(generations=generations)

    @property
    def _llm_type(self) -> str:
        return "mock-openai"
