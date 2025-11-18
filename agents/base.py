from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class AgentState:
    task: str
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseAgent(ABC):
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        
        self.name = name
        self.config = config or {}
        self.state = None
        self.logger = None  
        
    def set_logger(self, logger):
        self.logger = logger
        
    def log(self, message: str, level: str = "INFO"):
        if self.logger:
            self.logger.log(f"[{self.name}] {message}", level)
        else:
            print(f"[{self.name}] {message}")
    
    @abstractmethod
    def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentState:
        pass
    
    def validate_input(self, task: str, context: Optional[Dict[str, Any]] = None) -> bool:
        if not task or not isinstance(task, str):
            return False
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "config": self.config,
            "state": self.state.__dict__ if self.state else None
        }

