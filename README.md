# Motley crew of models

This is supposed to be an assortment of small toy stuff that I fancied to build to play around with a bit.

Nothing too serious, nothing production-ready and nothing really of too much interest to other parties.

# Contents

## Shape detection with X (Y)

A generator for shapes (e.g. lines, crosses, ...) in various colours which gets hooked up with a shape connector to solve the classification problem of identifying the shapes.
The thing to note here is that all data is generated on the fly and throw-away, i.e. the model will ever only see the data once.

## Time series prediction with LSTM (Pytorch Lightning)

A notebook, preferrably run on Colab to make use of their free GPU instances (TODO: Insert colab-launchable link to notebook.).

Creates a synthetic time series in the form of a Markov-series and then sets up an LSTM-model to learn and try to predict these. Very standard stuff, was just curious how well it would work with larger delays.

