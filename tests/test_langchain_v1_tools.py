import unittest
import ast
import operator
from functools import lru_cache

# 导入所需的LangChain组件
from langchain_core.tools import Tool, StructuredTool, tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
from pydantic import BaseModel, Field, field_validator
from langgraph.checkpoint.memory import MemorySaver
from mock_llm import MockChatOpenAI

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval_arithmetic(expression: str) -> float:
    """Evaluate a basic arithmetic expression without executing Python code."""
    def eval_node(node):
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](eval_node(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[type(node.op)](
                eval_node(node.left), eval_node(node.right)
            )
        raise ValueError("表达式只能包含数字和基础四则运算")

    return eval_node(ast.parse(expression, mode="eval"))

class TestLangChainV1Tools(unittest.TestCase):
    """测试LangChain v1.0+ Tools模块"""
    
    def setUp(self):
        """设置测试环境"""
        self.llm = MockChatOpenAI()
    
    def test_basic_tool(self):
        """测试基本工具"""
        def get_weather(city: str) -> str:
            """获取指定城市的天气信息"""
            weather_data = {
                "北京": "晴，25℃",
                "上海": "多云，23℃",
                "广州": "阴，28℃"
            }
            return weather_data.get(city, "未知城市")
        
        # 创建工具实例
        weather_tool = Tool(
            name="get_weather",
            func=get_weather,
            description="获取指定城市的天气信息，参数为城市名称"
        )
        
        # 测试工具运行
        result = weather_tool.run("北京")
        self.assertEqual(result, "晴，25℃")
        
        result = weather_tool.run("深圳")
        self.assertEqual(result, "未知城市")
    
    def test_structured_tool(self):
        """测试结构化工具"""
        # 定义参数结构
        class WeatherRequest(BaseModel):
            city: str = Field(description="要查询天气的城市名称")
            days: int = Field(default=1, description="要查询的天数")
        
        def get_weather_detailed(city: str, days: int = 1) -> str:
            """获取指定城市的详细天气信息"""
            weather_data = {
                "北京": ["晴，25℃", "多云，23℃", "阴，22℃"],
                "上海": ["多云，23℃", "阴，22℃", "小雨，20℃"],
                "广州": ["阴，28℃", "小雨，26℃", "多云，27℃"]
            }
            if city in weather_data:
                return "\n".join(weather_data[city][:days])
            return "未知城市"
        
        # 创建结构化工具
        weather_tool = StructuredTool.from_function(
            func=get_weather_detailed,
            name="get_weather_detailed",
            description="获取指定城市的详细天气信息",
            args_schema=WeatherRequest
        )
        
        # 测试工具运行
        result = weather_tool.run({"city": "北京", "days": 2})
        self.assertIn("晴，25℃", result)
        self.assertIn("多云，23℃", result)
    
    def test_tool_with_validation(self):
        """测试带参数验证的工具"""
        # 定义带验证的参数模型
        class CalculateRequest(BaseModel):
            expression: str = Field(description="要计算的数学表达式")
            
            @field_validator('expression')
            def validate_expression(cls, v):
                # 简单的表达式验证
                allowed_chars = set("0123456789+-*/() ")
                if not all(c in allowed_chars for c in v):
                    raise ValueError("表达式只能包含数字、运算符和括号")
                return v
        
        def calculate(expression: str) -> str:
            """计算数学表达式"""
            try:
                result = safe_eval_arithmetic(expression)
                return f"计算结果：{result}"
            except Exception as e:
                return f"计算错误：{str(e)}"
        
        # 创建结构化工具
        calculate_tool = StructuredTool.from_function(
            func=calculate,
            args_schema=CalculateRequest,
            description="计算数学表达式"
        )
        
        # 测试工具运行
        result = calculate_tool.run({"expression": "3+5*2"})
        self.assertIn("计算结果：13", result)
    
    def test_tool_with_llm(self):
        """测试工具与LLM集成"""
        def get_weather(city: str) -> str:
            """获取指定城市的天气信息"""
            weather_data = {
                "北京": "晴，25℃",
                "上海": "多云，23℃",
                "广州": "阴，28℃"
            }
            return weather_data.get(city, "未知城市")
        
        # 创建工具
        weather_tool = Tool(
            name="get_weather",
            func=get_weather,
            description="获取指定城市的天气信息，参数为城市名称"
        )
        
        # 绑定工具到模型
        tool_llm = self.llm.bind_tools([weather_tool])
        
        # 构建链
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个助手，需要使用工具来完成任务。"),
            ("human", "北京今天的天气怎么样？")
        ])
        
        chain = prompt | tool_llm
        response = chain.invoke({})
        
        # 验证响应包含工具调用
        self.assertTrue(hasattr(response, "tool_calls"))
        self.assertGreater(len(response.tool_calls), 0)
    
    def test_async_tool(self):
        """测试异步工具"""
        def sync_get_weather(city: str) -> str:
            """同步获取指定城市的天气信息"""
            weather_data = {
                "北京": "晴，25℃",
                "上海": "多云，23℃",
                "广州": "阴，28℃"
            }
            return weather_data.get(city, "未知城市")
        
        # 创建同步工具
        weather_tool = Tool(
            name="get_weather",
            func=sync_get_weather,
            description="获取指定城市的天气信息，参数为城市名称"
        )
        
        # 测试工具运行
        result = weather_tool.run("北京")
        self.assertEqual(result, "晴，25℃")
    
    def test_tool_error_handling(self):
        """测试工具错误处理"""
        def safe_calculate(expression: str) -> str:
            """安全计算数学表达式"""
            try:
                result = safe_eval_arithmetic(expression)
                return f"计算结果：{result}"
            except ValueError:
                return "表达式不安全"
            except Exception as e:
                return f"计算错误：{str(e)}"
        
        # 创建工具
        calculate_tool = Tool(
            name="safe_calculate",
            func=safe_calculate,
            description="安全计算数学表达式"
        )
        
        # 测试正常计算
        result = calculate_tool.run("3+5")
        self.assertIn("计算结果：8", result)
        
        # 测试错误处理
        result = calculate_tool.run("3/0")
        self.assertIn("计算错误", result)
        
        # 测试安全验证
        result = calculate_tool.run("__import__('os')")
        self.assertEqual(result, "表达式不安全")
    
    def test_tool_decorator(self):
        """测试@tool装饰器"""
        @tool
        def search_order(order_id: str) -> str:
            """根据订单ID查询快递状态。"""
            # 模拟数据库查询
            return f"订单 {order_id} 正在派送中..."
        
        # 测试工具运行
        result = search_order.run("12345")
        self.assertIn("订单 12345 正在派送中", result)
    
    def test_agent_integration(self):
        """测试Agent与工具集成"""
        def get_weather(city: str) -> str:
            """获取指定城市的天气信息"""
            weather_data = {
                "北京": "晴，25℃",
                "上海": "多云，23℃",
                "广州": "阴，28℃"
            }
            return weather_data.get(city, "未知城市")
        
        def calculate(expression: str) -> str:
            """计算数学表达式，如'1+1'、'2*3'等"""
            try:
                result = safe_eval_arithmetic(expression)
                return f"计算结果：{result}"
            except:
                return "计算错误"
        
        # 创建工具列表
        tools = [
            Tool(
                name="get_weather",
                func=get_weather,
                description="获取指定城市的天气信息，参数为城市名称"
            ),
            Tool(
                name="calculate",
                func=calculate,
                description="计算数学表达式，参数为数学表达式字符串"
            )
        ]
        
        # 创建Agent
        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt="你是一个助手，需要根据用户请求选择合适的工具来完成任务。"
        )
        
        # 测试Agent运行
        result = agent.invoke({
            "messages": [("human", "北京今天的天气怎么样？")]
        })
        
        # 验证结果包含天气信息
        self.assertIn("messages", result)
        self.assertGreater(len(result["messages"]), 0)
    
    def test_tool_caching(self):
        """测试工具缓存"""
        # 使用lru_cache缓存函数结果
        @lru_cache(maxsize=128)
        def get_weather(city: str) -> str:
            """获取指定城市的天气信息"""
            print(f"调用天气API获取{city}的天气")
            weather_data = {
                "北京": "晴，25℃",
                "上海": "多云，23℃",
                "广州": "阴，28℃"
            }
            return weather_data.get(city, "未知城市")
        
        # 创建工具
        weather_tool = Tool(
            name="get_weather",
            func=get_weather,
            description="获取指定城市的天气信息，参数为城市名称"
        )
        
        # 第一次调用（会执行函数）
        result1 = weather_tool.run("北京")
        self.assertEqual(result1, "晴，25℃")
        
        # 第二次调用（会使用缓存）
        result2 = weather_tool.run("北京")
        self.assertEqual(result2, "晴，25℃")
        
        # 验证缓存生效
        self.assertEqual(get_weather.cache_info().hits, 1)
    
    def test_tool_description_best_practices(self):
        """测试工具描述最佳实践"""
        def get_weather(city: str) -> str:
            """获取指定城市的天气信息"""
            return f"{city}的天气是晴天"
        
        # 创建详细描述的工具
        weather_tool = Tool(
            name="get_weather",
            func=get_weather,
            description="获取指定城市的天气信息。参数：city（城市名称，如'北京'、'上海'）。返回值：天气描述字符串，如'晴，25℃'。示例：get_weather('北京')"
        )
        
        # 测试工具运行
        result = weather_tool.run("北京")
        self.assertEqual(result, "北京的天气是晴天")
    
    def test_tool_with_checkpointer(self):
        """测试带检查点的Agent"""
        def get_weather(city: str) -> str:
            """获取指定城市的天气信息"""
            return f"{city}的天气是晴天"
        
        # 创建工具
        weather_tool = Tool(
            name="get_weather",
            func=get_weather,
            description="获取指定城市的天气信息"
        )
        
        # 创建内存检查点
        memory = MemorySaver()
        
        # 创建带检查点的Agent
        agent = create_agent(
            model=self.llm,
            tools=[weather_tool],
            checkpointer=memory,
            system_prompt="你是一个助手，需要使用工具来完成任务。"
        )
        
        # 测试带会话ID的调用
        config = {"configurable": {"thread_id": "session_001"}}
        result = agent.invoke(
            {"messages": [("human", "北京的天气怎么样？")]}, 
            config
        )
        
        # 验证结果
        self.assertIn("messages", result)

if __name__ == "__main__":
    unittest.main()
