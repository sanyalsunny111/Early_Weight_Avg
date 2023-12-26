import torch
import torch.nn as nn
import copy


def exists(val):
    return val is not None


class SWA(nn.Module):
    def __init__(
            self,
            model,
            swa_model=None,
            device=None,
            # if your model has lazylinears or other types of non-deepcopyable modules, you can pass in your own swa model
            update_after_step=100,
            update_every=10,
            param_or_buffer_names_no_swa=set(),
            ignore_names=set(),
            ignore_startswith_names=set(),
            include_online_model=True
            # set this to False if you do not wish for the online model to be saved along with the swa model (managed externally)
    ):
        super().__init__()

        # whether to include the online model within the module tree, so that state_dict also saves it

        self.include_online_model = include_online_model

        if include_online_model:
            self.online_model = model
        else:
            self.online_model = [model]  # hack

        # swa model

        self.swa_model = swa_model

        if not exists(self.swa_model):

            self.swa_model = copy.deepcopy(model)
            self.swa_model.eval()

        self.swa_model.requires_grad_(False)

        self.parameter_names = {name for name, param in self.swa_model.named_parameters() if
                                param.dtype in [torch.float, torch.float16]}
        self.buffer_names = {name for name, buffer in self.swa_model.named_buffers() if
                             buffer.dtype in [torch.float, torch.float16]}

        self.update_every = update_every
        self.update_after_step = update_after_step

        self.n_models = torch.tensor(0, dtype=torch.long, device=device)
        # self.n_models = 0

        assert isinstance(param_or_buffer_names_no_swa, (set, list))
        self.param_or_buffer_names_no_swa = param_or_buffer_names_no_swa  # parameter or buffer

        self.ignore_names = ignore_names
        self.ignore_startswith_names = ignore_startswith_names

        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('step', torch.tensor([0]))

    @property
    def model(self):
        return self.online_model if self.include_online_model else self.online_model[0]

    def restore_swa_model_device(self):
        device = self.initted.device
        self.swa_model.to(device)

    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    def copy_params_from_model_to_swa(self):
        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.swa_model),
                                                       self.get_params_iter(self.model)):
            ma_params.data.copy_(current_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.swa_model),
                                                         self.get_buffers_iter(self.model)):
            ma_buffers.data.copy_(current_buffers.data)

    def get_current_decay(self):
        decay = self.n_models / (self.n_models + 1)
        return decay

    def update(self):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_swa()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_swa()
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.swa_model, self.model)

        self.n_models += 1

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(self.get_params_iter(current_model),
                                                          self.get_params_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_swa:
                ma_params.data.copy_(current_params.data)
                continue

            ma_params.data.lerp_(current_params.data, 1. - current_decay)

        for (name, current_buffer), (_, ma_buffer) in zip(self.get_buffers_iter(current_model),
                                                          self.get_buffers_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_swa:
                ma_buffer.data.copy_(current_buffer.data)
                continue

            ma_buffer.data.lerp_(current_buffer.data, 1. - current_decay)

    def __call__(self, *args, **kwargs):
        return self.swa_model(*args, **kwargs)
