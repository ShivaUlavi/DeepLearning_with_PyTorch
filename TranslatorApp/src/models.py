import torch
import torch.nn as nn
import torch.nn.functional as F

from src.translation_utils import device, MAX_LENGTH


class EncoderRNN(nn.Module):
    """GRU Encoder that embeds English token indices and updates the recurrent hidden state."""

    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # input_size corresponds to the English vocabulary size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_tensor, hidden_tensor):
        # input_tensor shape: (batch_size, sequence_length) -> (1, 1) per step
        embedded = self.embedding(input_tensor)
        output, hidden_tensor = self.gru(embedded, hidden_tensor)
        return output, hidden_tensor

    def init_hidden(self, target_device=device):
        return torch.zeros(1, 1, self.hidden_size, device=target_device)


class AttnDecoderRNN(nn.Module):
    """Bahdanau-style Attention GRU Decoder that aligns French generation with Encoder states."""

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # output_size corresponds to the French vocabulary size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_tensor, hidden_tensor, encoder_outputs):
        embedded = self.embedding(input_tensor)
        embedded = self.dropout(embedded)

        # Calculate attention weights across the source sequence positions
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden_tensor[0]), 1)), dim=1
        )

        # Multiply attention weights by encoder outputs to compute the context vector
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(1), encoder_outputs.unsqueeze(0)
        )

        # Concatenate embedded input token and context vector, then project to hidden_size
        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden_tensor = self.gru(output, hidden_tensor)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden_tensor, attn_weights
