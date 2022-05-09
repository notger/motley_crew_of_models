# Motley crew of models

This is supposed to be an assortment of small toy stuff that I fancied to build to play around with a bit.

Nothing too serious, nothing production-ready and nothing really of too much interest to other parties.

# Contents

## Shape detection with a CNN

A generator for shapes (e.g. lines, crosses, ...) in various colours which gets hooked up with a shape connector to solve the classification problem of identifying the shapes.
The thing to note here is that all data is generated on the fly and throw-away, i.e. the model will ever only see the data once.

Usage: 
- Start TensorBoard with `tensorboard --logdir ./lightning_logs`
- Start the training with `python3 trainer.py`

If you want to change the batch sizes or other hyperparameters, then `trainer.py` is your friend.

## Time series prediction with LSTM (Pytorch Lightning)

A notebook, preferrably run on Colab to make use of their free GPU instances.
Use this link to launch the notebook directly from Github (needs a Google account, though) and do not forget to set the runtime to GPU: https://colab.research.google.com/github/notger/motley_crew_of_models/blob/main/timeseries_pred_lstm/demo_lstm_regression_for_timeseries.ipynb .

Creates a synthetic time series in the form of a Markov-series and then sets up an LSTM-model to learn and try to predict these. Very standard stuff, was just curious how well it would work with larger delays.

