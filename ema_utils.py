class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self._init_from_model(model)

    def _init_from_model(self, model):
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name] = value.detach().clone()
            else:
                self.shadow[name] = value.detach().clone()

    def update(self, model):
        for name, value in model.state_dict().items():
            if name not in self.shadow:
                self.shadow[name] = value.detach().clone()
                continue
            if not value.is_floating_point():
                self.shadow[name] = value.detach().clone()
                continue
            self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state.get("decay", self.decay)
        self.shadow = state.get("shadow", self.shadow)

    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=False)
