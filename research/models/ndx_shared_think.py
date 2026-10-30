"""Remove the stock-ID dense mixer; keep THINK's existing shared stock scorer."""
from ndx_extended_hours import ExtendedModel


class SharedThink(ExtendedModel):
    def __init__(self, nodes, groups, view):
        # Construct the matched old model first so unchanged parameters retain
        # exactly the same seed initialization, including the AH/PM branch.
        super().__init__(nodes, groups, view)
        self.core.kind = 'think'
        del self.core.market


def initialize_shared(model, family):
    if family != 'think':
        raise ValueError('This controlled ablation changes THINK only')
    assert not any(name.startswith('core.market.') for name, _ in model.named_parameters())
