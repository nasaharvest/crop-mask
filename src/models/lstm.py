import math

import torch
from torch import nn

from typing import Tuple, Optional


class UnrolledLSTM(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, dropout: float, batch_first: bool
    ) -> None:
        super().__init__()

        self.batch_first = batch_first
        self.hidden_size = hidden_size

        self.rnn = UnrolledLSTMCell(
            input_size=input_size, hidden_size=hidden_size, batch_first=batch_first
        )
        self.dropout = VariationalDropout(dropout)

    def forward(  # type: ignore
        self, x: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        sequence_length = x.shape[1] if self.batch_first else x.shape[0]
        batch_size = x.shape[0] if self.batch_first else x.shape[1]

        if state is None:
            # initialize to zeros
            hidden, cell = (
                torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size),
            )

            if x.is_cuda:
                hidden, cell = hidden.cuda(), cell.cuda()
        else:
            hidden, cell = state

        outputs = []
        for i in range(sequence_length):
            input_x = x[:, i, :].unsqueeze(1)
            _, (hidden, cell) = self.rnn(input_x, (hidden, cell))
            outputs.append(hidden)

            if self.training and (i == 0):
                self.dropout.update_mask(hidden.shape, hidden.is_cuda)

            hidden = self.dropout(hidden)

        return torch.stack(outputs, dim=0), (hidden, cell)


class UnrolledLSTMCell(nn.Module):
    """An unrolled LSTM, so that dropout can be applied between
    timesteps instead of between layers
    """

    def __init__(self, input_size: int, hidden_size: int, batch_first: bool) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first

        self.forget_gate = nn.Sequential(
            *[
                nn.Linear(
                    in_features=input_size + hidden_size, out_features=hidden_size, bias=True,
                ),
                nn.Sigmoid(),
            ]
        )

        self.update_gate = nn.Sequential(
            *[
                nn.Linear(
                    in_features=input_size + hidden_size, out_features=hidden_size, bias=True,
                ),
                nn.Sigmoid(),
            ]
        )

        self.update_candidates = nn.Sequential(
            *[
                nn.Linear(
                    in_features=input_size + hidden_size, out_features=hidden_size, bias=True,
                ),
                nn.Tanh(),
            ]
        )

        self.output_gate = nn.Sequential(
            *[
                nn.Linear(
                    in_features=input_size + hidden_size, out_features=hidden_size, bias=True,
                ),
                nn.Sigmoid(),
            ]
        )

        self.cell_state_activation = nn.Tanh()

        self.initialize_weights()

    def initialize_weights(self):
        sqrt_k = math.sqrt(1 / self.hidden_size)
        for parameters in self.parameters():
            for pam in parameters:
                nn.init.uniform_(pam.data, -sqrt_k, sqrt_k)

    def forward(  # type: ignore
        self, x: torch.Tensor, state: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        hidden, cell = state

        if self.batch_first:
            hidden, cell = torch.transpose(hidden, 0, 1), torch.transpose(cell, 0, 1)

        forget_state = self.forget_gate(torch.cat((x, hidden), dim=-1))
        update_state = self.update_gate(torch.cat((x, hidden), dim=-1))
        cell_candidates = self.update_candidates(torch.cat((x, hidden), dim=-1))

        updated_cell = (forget_state * cell) + (update_state * cell_candidates)

        output_state = self.output_gate(torch.cat((x, hidden), dim=-1))
        updated_hidden = output_state * self.cell_state_activation(updated_cell)

        if self.batch_first:
            updated_hidden = torch.transpose(updated_hidden, 0, 1)
            updated_cell = torch.transpose(updated_cell, 0, 1)

        return updated_hidden, (updated_hidden, updated_cell)


class VariationalDropout(nn.Module):
    """
    This ensures the same dropout is applied to each timestep,
    as described in https://arxiv.org/pdf/1512.05287.pdf
    """

    def __init__(self, p):
        super().__init__()

        self.p = p
        self.mask = None

    def update_mask(self, x_shape: Tuple, is_cuda: bool) -> None:
        mask = torch.bernoulli(torch.ones(x_shape) * (1 - self.p)) / (1 - self.p)
        if is_cuda:
            mask = mask.cuda()
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        return self.mask * x
