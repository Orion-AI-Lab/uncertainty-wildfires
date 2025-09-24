import torch.nn as nn
from models.modules.concrete_dropout import ConcreteDropout


def concrete_regularizer(model: nn.Module) -> nn.Module:
    def regularization(self) -> float:
        total_regularization = 0
        for module in filter(lambda x: isinstance(x, ConcreteDropout), self.modules()):
            total_regularization += module.regularization

        return total_regularization

    setattr(model, 'regularization', regularization)

    return model

