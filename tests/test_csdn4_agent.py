import unittest
import ast
import operator
from functools import lru_cache

# 导入所需的LangChain组件
from langchain_core.tools import tool, StructuredTool
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

class TestCsdn4Agent(unittest.TestCase):
    """测试csdn4_agent.md中的代码示例"""
    
    def setUp(self):
        """设置测试环境"""
        self.llm = MockChatOpenAI()
    
    def test_basic_tool_decorator(self):
        """测试基础工具定义（@tool装饰器）"""
        @tool
        def search_order(order_id: str) -> str:
            """根据订单ID查询快递状态。"""
            # 模拟数据库查询
            return f"订单 {order_id} 正在派送中..."
        
        # 测试工具运行
        result = search_order.run("12345")
        self.assertIn("订单 12345 正在派送中", result)
    
    def test_structured_tool(self):
        """测试结构化工具定义"""
        # 定义参数结构
        class WeatherRequest(BaseModel):
            city: str = Field(description="要查询天气的城市名称")
            days: int = Field(default=1, description="要查询的天数")
        
        def get_weather_detailed(city: str, days: int = 1) -> str:
            """获取指定城市的详细天气信息"""
            weather_data = {
                "北京": ["晴，25℃", "多云，23℃", "阴，22℃"],
                "上海": ["多云，23℃", "阴，22℃", "小雨，20℃"]
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
    
    def test_agent_creation(self):
        """测试新版Agent初始化"""
        @tool
        def search_order(order_id: str) -> str:
            """根据订单ID查询快递状态。"""
            # 模拟数据库查询
            return f"订单 {order_id} 正在派送中..."
        
        tools = [search_order]
        
        # 创建Agent
        agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt="你是一个专业的物流客服，必须通过工具查询信息。"
        )
        
        # 测试Agent运行
        result = agent.invoke({
            "messages": [("human", "帮我查一下订单号 12345 的状态")]
        })
        
        # 验证结果包含订单状态信息
        self.assertIn("messages", result)
        self.assertGreater(len(result["messages"]), 0)
    
    def test_agent_with_memory(self):
        """测试带记忆的Agent"""
        @tool
        def search_order(order_id: str) -> str:
            """根据订单ID查询快递状态。"""
            # 模拟数据库查询
            return f"订单 {order_id} 正在派送中..."
        
        tools = [search_order]
        
        # 创建内存检查点
        memory = MemorySaver()
        
        # 创建带记忆的Agent
        agent = create_agent(
            model=self.llm,
            tools=tools,
            checkpointer=memory,
            system_prompt="你是一个专业的物流客服，必须通过工具查询信息。"
        )
        
        # 测试带会话ID的调用
        config = {"configurable": {"thread_id": "session_001"}}
        result = agent.invoke(
            {"messages": [("human", "我是玄同")]}, 
            config
        )
        
        # 验证结果
        self.assertIn("messages", result)
        self.assertGreater(len(result["messages"]), 0)
    
    def test_tool_parameter_validation(self):
        """测试工具参数验证"""
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
        from langchain_core.tools import Tool
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
    
    def test_tool_performance_optimization(self):
        """测试工具性能优化"""
        # 使用lru_cache缓存函数结果
        @lru_cache(maxsize=100)
        def expensive_tool(query: str) -> str:
            # 复杂计算或API调用
            import time
            time.sleep(0.1)  # 模拟耗时操作
            return f"结果：{query}"
        
        # 创建工具
        from langchain_core.tools import Tool
        tool_instance = Tool(
            name="expensive_tool",
            func=expensive_tool,
            description="执行昂贵的操作"
        )
        
        # 第一次调用（会执行函数）
        result1 = tool_instance.run("test")
        self.assertEqual(result1, "结果：test")
        
        # 第二次调用（会使用缓存）
        result2 = tool_instance.run("test")
        self.assertEqual(result2, "结果：test")
        
        # 验证缓存生效
        self.assertEqual(expensive_tool.cache_info().hits, 1)

if __name__ == "__main__":
    unittest.main()
