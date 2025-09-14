model:
  name: "oso_lstm_predictor"
  type: "LSTM"
  layers:
    - type: "LSTM"
      units: 50
      return_sequences: true
    - type: "LSTM"
      units: 30
    - type: "Dense"
      units: 1
  compile:
    optimizer: "adam"
    loss: "mse"
    metrics: ["mae", "mse"]

training:
  epochs: 200
  batch_size: 32
  validation_split: 0.2
  early_stopping:
    patience: 25
    restore_best_weights: true

data:
  sequence_length: 12
  test_size: 0.2
  features:
    - "Jan"
    - "Feb"
    - "Mar"
    - "Apr"
    - "May"
    - "Jun"
    - "Jul"
    - "Aug"
    - "Sep"
    - "Oct"
    - "Nov"
    - "Dec"
  target: "среднее"