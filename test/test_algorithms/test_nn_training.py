from unittest.mock import patch

from demonstrator.neural_network import train_demonstrator_model

# todo: find a way to test without wandb tracking
#@patch('wandb.init')
#def test_train_nn():
#    train_demonstrator_model(n_epochs=2)