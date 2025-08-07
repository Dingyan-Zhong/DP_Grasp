import attr
from abc import ABC, abstractmethod

@attr.s
class DynamicValidationContext:
    grasp_pose = attr.ib()
   
@attr.s
class ValidationResult:
    is_valid = attr.ib()
    reason = attr.ib(default="")

class Rule(ABC):
    @abstractmethod
    def evaluate(self, context: DynamicValidationContext) -> ValidationResult:
        """Evaluates a grasp against a specific rule."""
        pass   