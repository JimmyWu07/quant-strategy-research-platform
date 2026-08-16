"""
v2 多因子选股系统 - 核心业务 Agent 包

统一导出所有 Agent 和 Coordinator,
使其他模块可通过 `from agents import AgentData` 直接引用.
"""

from agents.agent_data import AgentData
from agents.agent_factor import AgentFactor
from agents.agent_risk import AgentRisk
from agents.agent_macro import AgentMacro
from agents.agent_portfolio import AgentPortfolio
from agents.coordinator import Coordinator
