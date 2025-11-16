"""Base agent class for T4-OPT agent system."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class AgentState:
    """State container for agent execution."""
    task: str
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    """Base class for all T4-OPT agents."""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize base agent.
        
        Args:
            name: Agent identifier
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.state = None
        self.logger = None  # Will be set by agent manager
        
    def set_logger(self, logger):
        """Set logger instance."""
        self.logger = logger
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message."""
        if self.logger:
            self.logger.log(f"[{self.name}] {message}", level)
        else:
            print(f"[{self.name}] {message}")
    
    @abstractmethod
    def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentState:
        """
        Execute agent task.
        
        Args:
            task: Task description
            context: Optional context from previous agents
            
        Returns:
            AgentState with execution results
        """
        pass
    
    def validate_input(self, task: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Validate input before execution."""
        if not task or not isinstance(task, str):
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize agent state to dictionary."""
        return {
            "name": self.name,
            "config": self.config,
            "state": self.state.__dict__ if self.state else None
        }

