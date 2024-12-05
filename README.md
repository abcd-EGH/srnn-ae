# Outlier Detection for Time Series with Recurrent Autoencoder Ensembles (Torch Implementation)
This repository contains a PyTorch implementation of the paper "Outlier Detection for Time Series with Recurrent Autoencoder Ensembles" by Tung Kieu, Bin Yang, Chenjuan Guo, and Christian S. Jensen (IJCAI 2019). We referred to the code on https://github.com/tungk/OED. The goal of this project is to detect outliers in time series data using ensemble models based on recurrent autoencoders with several additional implementations: Dynamic Thresholding, Residual Connection, etc. You can find out what we experimented with outlier detection with our model on https://github.com/abcd-EGH/IEcapstone.

# Requirements
- Python 3.x
- Numpy
- Torch
- Pandas
- scikit-learn
- matplotlib
- seaborn
- arch

Install the required packages by running:
```bash
pip install -r requirements.txt
```

# Model
This implementation includes improved models with several additional features to improve outlier detection performance.
## Description of each Concept of Our Model
### Overview
Following three concepts represent different strategies to enhance model performance and efficiency:
- **Sparsely Connection** reduces model complexity by selectively activating connections, improving computational efficiency and preventing overfitting.
- **Residual Connection** facilitates training deep networks by allowing direct information flow from inputs to outputs, mitigating gradient vanishing issues.
- **Concatenation-based Skip (Encoder-Decoder) Connection** enhances sequence-to-sequence models by directly passing encoder information to the decoder, improving contextual understanding.
- **Variable-Skip Connection** allows the model to capture dependencies over varying time scales by systematically assigning different skip steps to each AutoEncoder in the ensemble. By distributing skip steps from 1 to N, the ensemble can effectively model both short-term and long-term dependencies.
- **Bi-directional LSTM** processes sequences in both forward and backward directions, providing the model with context from both past and future, thereby improving the overall understanding of the sequence.
- **Attention Mechanism** enables the model to focus on specific parts of the input when generating outputs, allowing it to handle longer sequences and complex alignments between input and output more effectively.
### Sparsely Connection
A method in artificial neural networks where only selected nodes are connected instead of all possible nodes. This reduces the model's complexity and improves computational efficiency. It helps prevent overfitting and enhances the model's generalization capabilities.<br>
**In the code**, the `sLSTMCell` class implements the sparsely connection. The key parts are found in the `masked_weight()` function and the `forward()` method.
- **Mask Generation (`masked_weight()` Function)**
  ```python
  def masked_weight(self):
      if self.step in self.mask_cache:
          return self.mask_cache[self.step]

      mask_combined = self.rng.randint(0, 3, size=self.hidden_size)
      masked_W1 = (mask_combined == 0) | (mask_combined == 2)
      masked_W2 = (mask_combined == 1) | (mask_combined == 2)

      masked_W1 = masked_W1.astype(np.float32)
      masked_W2 = masked_W2.astype(np.float32)

      self.mask_cache[self.step] = (masked_W1, masked_W2)

      tf_mask_W1 = torch.tensor(masked_W1, dtype=torch.float32, device=self.weight_h_2.device)
      tf_mask_W2 = torch.tensor(masked_W2, dtype=torch.float32, device=self.weight_h_2.device)
      return tf_mask_W1, tf_mask_W2
  ```
  - `mask_combined` generates random integers from 0 to 2 for each element in the hidden state size.
  - Depending on the value, `masked_W1` and `masked_W2` are generated to control the connections:
    - If the value is 0, the element is activated in `masked_W1`.
    - If the value is 1, it's activated in `masked_W2`.
    - If the value is 2, it's activated in both masks.
  - These masks are cached per time step to reuse the same mask.
- **Applying Masks (`forward()` Method)**
  ```python
  def forward(self, input, state):
    h, c = state
    self.step += 1

    new_h_1, new_c = self.lstm(input, (h, c))

    self.hidden_buffer.append(h.detach())
    if len(self.hidden_buffer) > self.skip_steps:
        h_skip = self.hidden_buffer.pop(0)
    else:
        h_skip = torch.zeros_like(h)

    new_h_2 = torch.sigmoid(torch.matmul(h_skip, self.weight_h_2) + self.bias_h_2)

    mask_w1, mask_w2 = self.masked_weight()

    new_h = new_h_1 * mask_w1 + new_h_2 * mask_w2

    new_state = (new_h, new_c)
    return new_h, new_state
  ```
  - `new_h_1` is the output from the LSTM cell based on the current input and previous state.
  - `new_h_2` is computed using a skip connection with the previous hidden state (`h_skip`).
  - `mask_w1` and `mask_w2` are the masks generated from `masked_weight()`, applied to `new_h_1` and `new_h_2`, respectively.
  - The final output `new_h` is a combination of the masked hidden states, achieving sparsely connected updates.

### Residual Connection
Residual connections are a technique used in deep neural networks to improve information flow and alleviate the vanishing gradient problem. By adding the input directly to the output, the model learns the "residual" of the desired mapping, facilitating easier learning, especially in very deep networks.<br>
**In the code**, residual connections are optionally implemented in the `sLSTMCell` class via the `use_cell_residual` parameter.
- **Initialization Stage**
  ```python
  def __init__(self, ..., use_cell_residual=False):
    ...
    self.use_cell_residual = use_cell_residual
    if self.use_cell_residual and input_size != hidden_size:
        self.cell_residual_transform = nn.Linear(input_size, hidden_size)
    else:
        self.cell_residual_transform = None
    ...
  ```
  - If `use_cell_residual` is `True` and the input size differs from the hidden state size, a linear layer `cell_residual_transform` is initialized to match dimensions.
- **Applying Residual Connection in `forward()` Method**
  ```python
  def forward(self, input, state):
    ...
    new_h = new_h_1 * mask_w1 + new_h_2 * mask_w2

    if self.use_cell_residual:
        if self.cell_residual_transform:
            cell_residual = self.cell_residual_transform(input)
        else:
            cell_residual = input
        new_h = new_h + cell_residual

    new_state = (new_h, new_c)
    return new_h, new_state
  ```
  - If residual connections are enabled, the input (`input`) is added to the current hidden state new_h.
  - If necessary, `cell_residual_transform` adjusts the input dimensions.
  - This addition allows the model to directly pass input information to the output, reducing information loss and improving gradient flow during training.

### Concatenation-based skip (Encoder-Decoder) Connection
In encoder-decoder architectures, concatenation-based skip connections involve directly passing intermediate or final hidden states from the encoder to the decoder. By concatenating the encoder's hidden states with the decoder's inputs, the decoder gains additional contextual information, which can improve performance, especially in sequence-to-sequence tasks.<br>
**In the code**, this concept is implemented in the `Decoder` class's `forward()` method, where the encoder's hidden states are concatenated with the decoder's inputs.
- **Adjusting Input Size in Decoder Initialization**
  ```python
  class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, ...):
        ...
        self.cells = nn.ModuleList([
            sLSTMCell(
                (output_size + hidden_size) if i == 0 else hidden_size,
                hidden_size,
                ...
            )
            for i in range(num_layers)
        ])
        ...
  ```
  - The input size for the first layer of the decoder is set to `output_size + hidden_size` to accommodate the concatenated input.
- **Concatenation in `forward()` Method**
  ```python
  def forward(self, targets, encoder_states, encoder_hidden_states):
    ...
    for t in range(seq_len):
        input_t = targets[t]
        encoder_hidden_t = encoder_hidden_states[t]

        input_t = torch.cat([input_t, encoder_hidden_t], dim=-1)

        for i, cell in enumerate(self.cells):
            h_i, c_i = states[i]
            h_i, (h_i, c_i) = cell(input_t, (h_i, c_i))
            states[i] = (h_i, c_i)
            input_t = h_i
        output_t = self.output_layer(input_t)
        outputs.append(output_t)
    ...
  ```
  - At each time step `t`, the decoder input `input_t` and the encoder hidden state `encoder_hidden_t` are concatenated.
  - This concatenated input is fed into the decoder's LSTM cells, allowing the decoder to utilize rich contextual information from the encoder.

### Variable-Skip Connection
Variable-Skip Connection refers to a mechanism where each component in an ensemble model uses a different skip-step size in its recurrent connections. In recurrent neural networks (RNNs), skip connections allow the model to connect non-adjacent time steps, effectively capturing dependencies over longer intervals. By assigning different skip steps to each AutoEncoder in the ensemble, the model can learn temporal dependencies at multiple scales simultaneously.<br>
**In the code**, Variable-Skip Connection is implemented in the ensemble classes like `EVSLAE`, where each AutoEncoder is assigned a unique `skip_steps` value ranging systematically from 1 to `N`, where `N` is the number of AutoEncoders in the ensemble.
- **Assignment of `skip_steps` in Ensemble Models**
 ```python
 class EVSLAE(nn.Module):
    def __init__(..., N, ..., **kwargs):
        ...
        for idx in range(N):
            # Set skip_steps from 1 to N for each AutoEncoder
            random_skip_steps = idx + 1
            ...
            autoencoder = AutoEncoder(
                ...,
                skip_steps=random_skip_steps,
                ...
            )
            self.autoencoders.append(autoencoder)
 ```
 - In the EVSLAE class, each AutoEncoder in the ensemble is assigned a skip_steps value equal to its index plus one, effectively distributing skip steps from 1 to `N`.
 - This systematic assignment ensures that the ensemble covers a range of temporal dependencies, from short-term to long-term.
- **Effect on Temporal Dependencies**
  - AutoEncoders with smaller `skip_steps` (e.g., 1) capture short-term dependencies by connecting nearby time steps.
  - AutoEncoders with larger `skip_steps` (e.g., `N`) capture longer-term dependencies by connecting more distant time steps.
  - By combining the outputs of all AutoEncoders, the ensemble effectively models a wide spectrum of temporal relationships in the data.
- **Skip Connection in `sLSTMCell`**
  ```python
  def forward(self, input, state):
    ...
    # Update hidden buffer
    self.hidden_buffer.append(h.detach())
    if len(self.hidden_buffer) > self.skip_steps:
        h_skip = self.hidden_buffer.pop(0)
    else:
        h_skip = torch.zeros_like(h)
    ...
    # Compute new_h_2 using skip connection
    new_h_2 = torch.sigmoid(torch.matmul(h_skip, self.weight_h_2) + self.bias_h_2)
    ...
  ```
  - The `sLSTMCell` uses the `skip_steps` value to determine which past hidden state to use for the skip connection.
  - The `hidden_buffer` stores past hidden states, and `h_skip` is obtained by popping the oldest hidden state after `skip_steps` time steps.
  - This mechanism allows each `sLSTMCell` to incorporate information from a specific time step in the past, determined by its `skip_steps` value.

### Bi-directional LSTM
A Bi-directional LSTM (Bi-LSTM) processes data in both forward and backward directions, allowing the model to have information from both past and future contexts. This is particularly useful in sequence modeling tasks where context from both ends of the sequence can improve performance.<br>
**In the code**, the `BidirectionalEncoder` class implements a bi-directional LSTM by maintaining separate forward and backward `sLSTMCell` layers. The `Encoder` class can instantiate either the original unidirectional encoder or the bidirectional encoder based on the `bidirectional` parameter.
- **Bidirectional Encoder Initialization**
  ```python
  class BidirectionalEncoder(nn.Module):
      def __init__(..., bidirectional=True, **kwargs):
          ...
          self.forward_cells = nn.ModuleList([...])
          self.backward_cells = nn.ModuleList([...])
          ...
  ```
  - The encoder initializes separate `forward_cells` and `backward_cells` for processing the sequence in both directions.
  - The input size for subsequent layers is adjusted accordingly.
- **Forward Pass in Both Directions**
  ```python
  def forward(self, inputs):
    ...
    for layer in range(self.num_layers):
        ...
        # Forward pass
        for t in range(seq_len):
            ...
            h_forward, (h_forward, c_forward) = self.forward_cells[layer](input_t, (h_forward, c_forward))
            forward_outputs.append(h_forward)
        # Backward pass
        for t in reversed(range(seq_len)):
            ...
            h_backward, (h_backward, c_backward) = self.backward_cells[layer](input_t, (h_backward, c_backward))
            backward_outputs.insert(0, h_backward)
        # Concatenate outputs
        layer_output = torch.cat((forward_outputs, backward_outputs), dim=2)
        ...
  ```
  - The model processes the input sequence in the forward direction using `forward_cells` and in the reverse direction using `backward_cells`.
  - The outputs from both directions are concatenated at each time step.
- **Adjustments in Decoder and AutoEncoder**
  ```python
  class Decoder(nn.Module):
    def __init__(..., bidirectional_encoder=True, **kwargs):
        ...
        decoder_hidden_size = hidden_size if not bidirectional_encoder else hidden_size * 2
        self.cells = nn.ModuleList([...])
        ...
  ```
  - The decoder adjusts the hidden size based on whether the encoder is bidirectional.
  - This ensures compatibility between the encoder's output and the decoder's input.
### Attention Mechanism
The attention mechanism enables the model to focus on specific parts of the input sequence when generating each part of the output sequence. It computes a weighted sum of encoder outputs, where the weights are learned based on how relevant each input is to the current output. This allows the model to handle long sequences and improves performance in tasks where alignment between input and output is crucial.<br>
**In the code**, the attention mechanism is implemented in the `Decoder` class of the last model. Specifically, the attention is calculated during the decoder's forward pass by computing scores between the decoder's current hidden state and the encoder's outputs.
- **Initialization of Attention Components**
  ```python
  class Decoder(nn.Module):
    def __init__(...):
        ...
        self.output_layer = nn.Linear(hidden_size * 2, output_size)
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.softmax = nn.Softmax(dim=1)
  ```
  - The `attention` linear layer computes the energy scores between the decoder hidden state and encoder outputs.
  - The `output_layer` is adjusted to accommodate the concatenated context vector and decoder hidden state.
- **Attention Computation in Forward Pass**
  ```python
  def forward(self, targets, encoder_hidden_states, encoder_final_states):
    ...
    encoder_outputs = encoder_hidden_states[-1]
    encoder_outputs = encoder_outputs.transpose(0, 1)  # (batch, seq_len, hidden)
    ...
    for t in range(seq_len):
        ...
        # Attention Mechanism
        h_i_expanded = h_i.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        energy = torch.tanh(self.attention(torch.cat((h_i_expanded, encoder_outputs), dim=2)))
        scores = torch.sum(energy, dim=2)
        attn_weights = self.softmax(scores)
        # Compute context vector
        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        context = context.squeeze(1)
        # Combine context with decoder hidden state
        combined = torch.cat((h_i, context), dim=1)
        combined = torch.tanh(combined)
        # Generate output
        output_t = self.output_layer(combined)
        outputs.append(output_t)
  ```
  - Energy Calculation
    - The decoder hidden state `h_i` is expanded and concatenated with encoder outputs.
    - The concatenated tensor passes through a linear layer and activation function to compute the energy scores.
  - Attention Weights
    - The energy scores are summed along the hidden dimension and passed through a softmax to obtain attention weights.
  - Context Vector
    - The attention weights are used to compute a weighted sum of the encoder outputs, resulting in the context vector.
  - Combining Context and Hidden State
    - The context vector is concatenated with the decoder hidden state and passed through an activation function.
    - This combined vector is then used to generate the output.
## Base Model
- `BLAE`: Bi-directional LSTM and AutoEncoder (No Sparsely Connection)
- `ESLAE`: Ensemble of **Sparsely Connection** LSTM and AutoEncoder

## Advanced Model
- `ERSLAE`: Ensemble of **Residual** & Sparsely Connection LSTM and AutoEncoder
- `ECSLAE`: Ensemble of **Concatenation-based skip (Encoder-Decoder)** & Sparsely Connection LSTM and AutoEncoder
- `EVSLAE`: Ensemble of **Variable skip** & Sparsely Connection LSTM and AutoEncoder
- `ESBLAE`: Ensemble of Sparsely Connection **Bi-directional LSTM** and AutoEncoder
- `EASLAE`: Ensemble of **Attention** & Sparsely Connection LSTM and AutoEncoder

# Installation
```bash
pip install git+https://github.com/abcd-EGH/srnn-ae
```

# Citation
If you find this code or project useful in your research, please cite the original paper:
```
@inproceedings{tungbcc19,
  title={Outlier Detection for Time Series with Recurrent Autoencoder Ensembles},
  author={Kieu, Tung and Yang, Bin and Guo, Chenjuan and S. Jensen, Christian},
  booktitle={International Joint Conference on Artificial Intelligence (IJCAI '19)},
  year={2019}
}
```