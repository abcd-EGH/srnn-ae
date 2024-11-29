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

### Residual Connection
Residual connections are a technique used in deep neural networks to improve information flow and alleviate the vanishing gradient problem. By adding the input directly to the output, the model learns the "residual" of the desired mapping, facilitating easier learning, especially in very deep networks.<br>
**In the code**, residual connections are optionally implemented in the sLSTMCell class via the use_cell_residual parameter.
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

### Concatenation-based skip (Encoder-Decoder) Connection:
In encoder-decoder architectures, concatenation-based skip connections involve directly passing intermediate or final hidden states from the encoder to the decoder. By concatenating the encoder's hidden states with the decoder's inputs, the decoder gains additional contextual information, which can improve performance, especially in sequence-to-sequence tasks.<br>
**In the code**, this concept is implemented in the Decoder class's forward() method, where the encoder's hidden states are concatenated with the decoder's inputs.
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
## Base Model
- BLAE: Bi-directional LSTM and AutoEncoder (No Sparsely Connection)
- ESLAE: Ensemble of Sparsely Connection LSTM and AutoEncoder

## Advanced Model
- ERSLAE: Ensemble of Residual & Sparsely Connection LSTM and AutoEncoder
- ECSLAE: Ensemble of Concatenation-based skip (Encoder-Decoder) & Sparsely Connection LSTM and AutoEncoder
- EVSLAE: Ensemble of Variable skip & Sparsely Connection LSTM and AutoEncoder

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