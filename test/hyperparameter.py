def hyperparameter_setting(**kwargs):
    """
    <Outlier Detection for Time Series with Recurrent Autoencoder Ensembles>
    [Hyperparameters Settings]
    For all deep learning based methods, we use
    Adadelta [Zeiler, 2012] as the optimizer, and we set
    their learning rates to 10e-3.
    ...
    we set the number of hidden LSTM units to 8;
    we set the default number of autoencoders N to
    40, and we also study the effect of varying N from 10 to 40;
    and we set λ to 0.005.
    We randomly vary the skip connection jump step size L from 1 to 10.
    ...
    For MP*, we set the pattern size to 10.
    * I think MP is Markov Process, and the pattern size is the state space of MP.
    """
    args = {}
    args['N'] = kwargs['N'] if 'N' in kwargs else 10 # 앙상블 모델 수, 10~40
    args['input_size'] = kwargs['input_size'] if 'input_size' in kwargs else 1  # 단일 시계열
    args['hidden_size'] = kwargs['hidden_size'] if 'hidden_size' in kwargs else 8
    args['output_size'] = kwargs['output_size'] if 'output_size' in kwargs else 1
    args['num_layers'] = kwargs['num_layers'] if 'num_layers' in kwargs else 1 # No mention in the paper
    args['limit_skip_steps'] = kwargs['limit_skip_steps'] if 'limit_skip_steps' in kwargs else 10 # L: 1~10 랜덤
    args['learning_rate'] = kwargs['learning_rate'] if 'learning_rate' in kwargs else 1e-3
    args['l1_lambda'] = kwargs['l1_lambda'] if 'l1_lambda' in kwargs else 0.005
    args['batch_size'] = kwargs['batch_size'] if 'batch_size' in kwargs else 32
    args['window_size'] = kwargs['window_size'] if 'window_size' in kwargs else 10
    args['num_epochs'] = kwargs['num_epochs'] if 'num_epochs' in kwargs else 100

    return args

if __name__ == '__main__':
    args = hyperparameter_setting(N=10)
    print(args)


