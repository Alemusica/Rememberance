"""
Agents - Sistema Nervoso dell'evoluzione (LLM supervisors)
"""

from .base import BaseAgent, AgentResponse
from .orchestrator import OrchestratorAgent

__all__ = ["BaseAgent", "AgentResponse", "OrchestratorAgent"]
