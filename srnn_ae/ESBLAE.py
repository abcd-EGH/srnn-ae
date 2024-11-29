import numpy as np
import torch
import torch.nn as nn
import random
# Definition of sLSTMCell
class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, forget_bias=1.0, dense=None,
                 file_name='h1', type='enc', component=1, partition=1, seed=None, skip_steps=1):
        super(sLSTMCell, self).__init__()
        self.hidden_size = hidden_size
        self.forget_bias = forget_bias
        self.file_name = file_name
        self.type = type
        self.component = component
        self.partition = partition
        self.step = 0  # Equivalent to TensorFlow's _step
        self.skip_steps = skip_steps  # Added skip_steps

        # Initialize the main LSTM cell
        self.lstm = nn.LSTMCell(input_size, hidden_size)

        # Initialize weights of LSTM cell with Xavier initialization
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # Initialize additional weights and biases
        self.weight_h_2 = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.bias_h_2 = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.xavier_normal_(self.weight_h_2)
        nn.init.zeros_(self.bias_h_2)

        # Activation function is now Tanh
        self.activation = torch.tanh

        # Initialize mask generator with seed for reproducibility
        if seed is not None:
            self.seed = seed
            self.rng = np.random.RandomState(seed)
        else:
            self.seed = None
            self.rng = np.random.RandomState()

        # Buffer to store past hidden states for skip connection
        self.hidden_buffer = []

        # Initialize mask cache
        self.mask_cache = {}  # step: (mask_w1, mask_w2)

    def masked_weight(self):
        # Generate masks without saving or loading from files
        if self.step in self.mask_cache:
            return self.mask_cache[self.step]

        # Generate new masks and cache them
        mask_combined = self.rng.randint(0, 3, size=self.hidden_size)
        masked_W1 = (mask_combined == 0) | (mask_combined == 2)
        masked_W2 = (mask_combined == 1) | (mask_combined == 2)

        masked_W1 = masked_W1.astype(np.float32)
        masked_W2 = masked_W2.astype(np.float32)

        self.mask_cache[self.step] = (masked_W1, masked_W2)

        # Convert masks to torch tensors
        masked_W1, masked_W2 = self.mask_cache[self.step]
        tf_mask_W1 = torch.tensor(masked_W1, dtype=torch.float32, device=self.weight_h_2.device)
        tf_mask_W2 = torch.tensor(masked_W2, dtype=torch.float32, device=self.weight_h_2.device)
        return tf_mask_W1, tf_mask_W2

    def forward(self, input, state):
        """
        Args:
            input: Tensor of shape (batch_size, input_size)
            state: Tuple of (h, c), each of shape (batch_size, hidden_size)
        Returns:
            h: New hidden state
            new_state: Tuple of (new_h, new_c)
        """
        h, c = state
        self.step += 1

        # Compute LSTM cell output
        new_h_1, new_c = self.lstm(input, (h, c))

        # Update hidden buffer
        self.hidden_buffer.append(h.detach())
        if len(self.hidden_buffer) > self.skip_steps:
            h_skip = self.hidden_buffer.pop(0)
        else:
            h_skip = torch.zeros_like(h)  # Initialize with zeros if not enough past steps

        # Compute new_h_2 using skip connection with Sigmoid activation
        new_h_2 = torch.sigmoid(torch.matmul(h_skip, self.weight_h_2) + self.bias_h_2)

        # Get masks
        mask_w1, mask_w2 = self.masked_weight()

        # Apply masks with (1,0), (0,1), (1,1) mapping
        new_h = new_h_1 * mask_w1 + new_h_2 * mask_w2

        new_state = (new_h, new_c)
        return new_h, new_state

    def reset_step(self):
        """Reset the step counter and mask cache."""
        self.step = 0
        self.mask_cache = {}

# Original (Unidirectional) Encoder
class OriginalEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, skip_steps=1, file_name='enc', partition=1, **kwargs):
        super(OriginalEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Use ModuleList to store multiple layers of sLSTMCell
        self.cells = nn.ModuleList([
            sLSTMCell(
                input_size if i == 0 else hidden_size,
                hidden_size,
                type='enc',
                component=i+1,
                partition=partition,
                file_name=file_name,
                skip_steps=skip_steps,
                **kwargs
            )
            for i in range(num_layers)
        ])

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape (seq_len, batch_size, input_size)
        Returns:
            outputs: List of final hidden states for each layer
            states: List of final (h, c) tuples for each layer
        """
        batch_size = inputs.size(1)
        seq_len = inputs.size(0)

        # Initialize hidden and cell states to zeros
        h = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) for _ in range(self.num_layers)]
        states = list(zip(h, c))

        # Iterate over each time step
        for t in range(seq_len):
            input_t = inputs[t]
            for i, cell in enumerate(self.cells):
                h_i, c_i = states[i]
                h_i, (h_i, c_i) = cell(input_t, (h_i, c_i))
                states[i] = (h_i, c_i)
                input_t = h_i  # Use current layer's output as next layer's input
        outputs = [state[0] for state in states]  # Final hidden states of each layer
        return outputs, states

# Bidirectional Encoder
class BidirectionalEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, skip_steps=1, file_name='enc', partition=1, **kwargs):
        super(BidirectionalEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.file_name = file_name
        self.partition = partition

        # Forward and Backward cells for each layer
        self.forward_cells = nn.ModuleList([
            sLSTMCell(
                input_size if i == 0 else hidden_size * 2,  # Increased input size due to bidirectional concatenation
                hidden_size,
                type='enc',
                component=i+1,
                partition=partition,
                file_name=f'{file_name}_forward_layer{i+1}',
                skip_steps=skip_steps,
                **kwargs
            )
            for i in range(num_layers)
        ])
        self.backward_cells = nn.ModuleList([
            sLSTMCell(
                input_size if i == 0 else hidden_size * 2,
                hidden_size,
                type='enc',
                component=i+1,
                partition=partition,
                file_name=f'{file_name}_backward_layer{i+1}',
                skip_steps=skip_steps,
                **kwargs
            )
            for i in range(num_layers)
        ])

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape (seq_len, batch_size, input_size)
        Returns:
            outputs: List of final hidden states for each layer (forward and backward concatenated)
            states: List of final (h, c) tuples for each layer and direction
        """
        batch_size = inputs.size(1)
        seq_len = inputs.size(0)

        # Initialize hidden and cell states for forward and backward
        h_forward = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) for _ in range(self.num_layers)]
        c_forward = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) for _ in range(self.num_layers)]
        states_forward = list(zip(h_forward, c_forward))

        h_backward = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) for _ in range(self.num_layers)]
        c_backward = [torch.zeros(batch_size, self.hidden_size, device=inputs.device) for _ in range(self.num_layers)]
        states_backward = list(zip(h_backward, c_backward))

        # Forward pass
        forward_outputs = []
        for t in range(seq_len):
            input_t = inputs[t]
            for i, cell in enumerate(self.forward_cells):
                h_i, c_i = states_forward[i]
                h_new, (h_i, c_i) = cell(input_t, (h_i, c_i))
                states_forward[i] = (h_new, c_i)
                input_t = h_new  # Input for next layer
            forward_outputs.append(input_t)

        # Backward pass
        backward_outputs = []
        for t in reversed(range(seq_len)):
            input_t = inputs[t]
            for i, cell in enumerate(self.backward_cells):
                h_i, c_i = states_backward[i]
                h_new, (h_i, c_i) = cell(input_t, (h_i, c_i))
                states_backward[i] = (h_new, c_i)
                input_t = h_new  # Input for next layer
            backward_outputs.insert(0, input_t)  # Insert at beginning to maintain order

        # Concatenate forward and backward outputs
        combined_outputs = []
        for f_out, b_out in zip(forward_outputs, backward_outputs):
            combined = torch.cat((f_out, b_out), dim=1)  # Shape: (batch_size, hidden_size*2)
            combined_outputs.append(combined)
        combined_outputs = torch.stack(combined_outputs, dim=0)  # Shape: (seq_len, batch_size, hidden_size*2)

        # Concatenate final states from forward and backward
        final_outputs = []
        final_states = []
        for i in range(self.num_layers):
            combined_hidden = torch.cat((states_forward[i][0], states_backward[i][0]), dim=1)  # (batch_size, hidden_size*2)
            final_outputs.append(combined_hidden)
            combined_state = (torch.cat((states_forward[i][0], states_backward[i][0]), dim=1),
                              torch.cat((states_forward[i][1], states_backward[i][1]), dim=1))
            final_states.append(combined_state)

        return final_outputs, final_states

# Encoder that can be bidirectional or unidirectional
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, skip_steps=1, file_name='enc', partition=1, bidirectional=True, **kwargs):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        if self.bidirectional:
            self.encoder = BidirectionalEncoder(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                skip_steps=skip_steps,
                file_name=file_name,
                partition=partition,
                **kwargs
            )
            self.output_size = hidden_size * 2  # Doubled due to bidirectionality
        else:
            self.encoder = OriginalEncoder(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                skip_steps=skip_steps,
                file_name=file_name,
                partition=partition,
                **kwargs
            )
            self.output_size = hidden_size

    def forward(self, inputs):
        """
        Args:
            inputs: Tensor of shape (seq_len, batch_size, input_size)
        Returns:
            outputs: List of final hidden states for each layer
            states: List of final (h, c) tuples for each layer
        """
        return self.encoder(inputs)

# Decoder class
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, skip_steps=1, file_name='dec', partition=1, bidirectional_encoder=False, seed=None, **kwargs):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.bidirectional_encoder = bidirectional_encoder

        # Initialize the Decoder's sLSTMCells with the adjusted hidden_size
        self.cells = nn.ModuleList([
            sLSTMCell(
                output_size if i == 0 else hidden_size,
                hidden_size,
                type='dec',
                component=i+1,
                partition=partition,
                file_name=f'{file_name}_layer{i+1}',
                skip_steps=skip_steps,
                seed=seed,
                **kwargs
            )
            for i in range(num_layers)
        ])
        
        # Final output layer remains the same
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, targets, encoder_states):
        """
        Args:
            targets: Tensor of shape (seq_len, batch_size, output_size)
            encoder_states: List of (h, c) tuples from the encoder
        Returns:
            outputs: Reconstructed outputs of shape (seq_len, batch_size, output_size)
        """
        batch_size = targets.size(1)
        seq_len = targets.size(0)

        # Initialize hidden and cell states with encoder's final states
        h = [state[0].detach() for state in encoder_states]
        c = [state[1].detach() for state in encoder_states]
        states = list(zip(h, c))

        outputs = []
        for t in range(seq_len):
            input_t = targets[t]
            for i, cell in enumerate(self.cells):
                h_i, c_i = states[i]
                h_i, (h_i, c_i) = cell(input_t, (h_i, c_i))
                states[i] = (h_i, c_i)
                input_t = h_i  # Pass to next layer
            output_t = self.output_layer(input_t)
            outputs.append(output_t)
        outputs = torch.stack(outputs, dim=0)
        return outputs

# AutoEncoder class that combines Encoder and Decoder
class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, skip_steps=1,
                 file_name_enc='enc', file_name_dec='dec', partition=1, bidirectional=True, seed=777, **kwargs):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            skip_steps=skip_steps,
            file_name=file_name_enc,
            partition=partition,
            bidirectional=bidirectional,
            seed=seed,
            **kwargs
        )
        
        # Adjust hidden_size for Decoder based on bidirectionality
        decoder_hidden_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.decoder = Decoder(
            hidden_size=decoder_hidden_size,  # Use adjusted hidden size
            output_size=output_size,
            num_layers=num_layers,
            skip_steps=skip_steps,
            file_name=file_name_dec,
            partition=partition,
            bidirectional_encoder=bidirectional,
            seed=seed,
            **kwargs
        )
    
    def forward(self, inputs, targets):
        encoder_outputs, encoder_states = self.encoder(inputs)
        outputs = self.decoder(targets, encoder_states)
        return outputs

# Ensemble of AutoEncoders (ESBLAE)
class ESBLAE(nn.Module):
    def __init__(self, N, input_size, hidden_size, output_size, num_layers=1, limit_skip_steps=2,
                 file_names=None, seed=777, bidirectional=True, **kwargs):
        """
        Args:
            N: Number of AutoEncoders in the ensemble
            input_size: Size of the input features
            hidden_size: Size of the hidden state in LSTM
            output_size: Size of the output features
            num_layers: Number of layers in each AutoEncoder
            limit_skip_steps: Maximum value for skip_steps (e.g., 2 means skip_steps can be 1 or 2)
            file_names: List of file_names for each AutoEncoder. Length should be N.
            bidirectional: Whether to use bidirectional encoder
            **kwargs: Additional keyword arguments for sRLSTMCell
        """
        super(ESBLAE, self).__init__()
        self.N = N
        self.autoencoders = nn.ModuleList()

        # Default file names if not provided
        if file_names is None:
            file_names = [f'model{i}' for i in range(N)]
        elif len(file_names) != N:
            raise ValueError("Length of file_names must be equal to N")

        for idx in range(N):
            # Randomly choose skip_steps from 1 to limit_skip_steps for each AutoEncoder
            random_skip_steps = random.choice([i for i in range(1, limit_skip_steps+1)])

            # Define encoder and decoder file names
            file_name_enc = f'{file_names[idx]}_enc'
            file_name_dec = f'{file_names[idx]}_dec'

            # Partition number
            partition = idx + 1

            # Set a unique seed for each AutoEncoder for reproducibility
            autoencoder_seed = seed + idx

            autoencoder = AutoEncoder(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                num_layers=num_layers,
                skip_steps=random_skip_steps,
                file_name_enc=file_name_enc,
                file_name_dec=file_name_dec,
                partition=partition,
                bidirectional=bidirectional,
                seed=autoencoder_seed,
                **kwargs
            )
            self.autoencoders.append(autoencoder)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Tensor of shape (seq_len, batch_size, input_size)
            targets: Tensor of shape (seq_len, batch_size, output_size)
        Returns:
            outputs: Averaged reconstructed outputs of shape (seq_len, batch_size, output_size)
        """
        ensemble_outputs = []
        for autoencoder in self.autoencoders:
            output = autoencoder(inputs, targets)
            ensemble_outputs.append(output)
        # Stack and average the outputs
        stacked_outputs = torch.stack(ensemble_outputs, dim=0)  # Shape: (N, seq_len, batch_size, output_size)
        averaged_output = torch.mean(stacked_outputs, dim=0)    # Shape: (seq_len, batch_size, output_size)
        return averaged_output

    def reset_steps(self):
        """Reset step counters and mask caches for all AutoEncoders in the ensemble."""
        for autoencoder in self.autoencoders:
            if autoencoder.encoder.bidirectional:
                for cell in autoencoder.encoder.encoder.forward_cells:
                    cell.reset_step()
                for cell in autoencoder.encoder.encoder.backward_cells:
                    cell.reset_step()
            else:
                for cell in autoencoder.encoder.encoder.cells:
                    cell.reset_step()
            for cell in autoencoder.decoder.cells:
                cell.reset_step()