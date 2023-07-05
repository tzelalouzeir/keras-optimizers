# Using Keras Optimizers with TensorFlow

This repository provides an example of how to use various optimizers available in Keras, which is integrated with TensorFlow, for training neural networks. The code demonstrates how to define a model using Keras and TensorFlow and optimize it using different optimizers.

## Getting Started

To use the optimizers in Keras with TensorFlow, you need to have both Keras and TensorFlow installed in your Python environment. You can install them using pip:

```shell
pip install tensorflow
pip install keras
```

You will also need other dependencies such as NumPy and scikit-learn to run the code in the example.

## Code Example

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import keras_tuner as kt

def model_builder(hp):
    model = Sequential()
    
    # Define hyperparameters
    hp_units_1 = hp.Int('units_1', min_value=32, max_value=128, step=32)
    hp_units_2 = hp.Int('units_2', min_value=32, max_value=128, step=32)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    hp_optimizer = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam'])
    
    # Build the model
    model.add(LSTM(units=hp_units_1, return_sequences=True, input_shape=(1, len(signals_predic))))
    model.add(Dropout(0.2))
    model.add(LSTM(units=hp_units_2))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    
    # Compile the model
    model.compile(optimizer=hp_optimizer, loss='mean_squared_error')

    return model

def yourModel():
    # Load and preprocess the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Set up hyperparameter tuning
    tuner = kt.Hyperband(model_builder, objective='val_loss', max_epochs=10, factor=3, directory='my_dir', project_name='intro_to_kt')
    stop_early = EarlyStopping(monitor='val_loss', patience=10)
    
    # Perform hyperparameter search
    tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])
    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Print the best hyperparameters
    print(f"""
        The hyperparameter search is complete. 
        The optimal number of units in the first LSTM layer is {best_hps.get('units_1')} 
        and the optimal number of units in the second LSTM layer is {best_hps.get('units_2')}.
        The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
        The best optimizer used is {best_hps.get('optimizer')}.
    """)

    # Build and train the model with the best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    
    # Return the trained model
    return model
```

## Usage

To use

 the code, follow these steps:

1. Install the required dependencies mentioned in the "Getting Started" section.
2. Copy the code into your Python environment or script.
3. Modify the code to fit your specific dataset and requirements.
4. Run the `yourModel` function to train the model using hyperparameter tuning.

## Optimizers

The code example demonstrates the usage of various optimizers available in Keras with TensorFlow. The hyperparameter `hp_optimizer` in the `model_builder` function allows you to choose from the following optimizers:

- Adam (default)
- Stochastic Gradient Descent (SGD)
- RMSprop
- Adagrad
- Adadelta
- Adamax
- Nadam

You can modify the `values` parameter in the `hp_optimizer` choice to include or exclude specific optimizers.

## Contributing

If you want to contribute to this repository, feel free to fork the project and submit a pull request with your changes. We appreciate any contributions and improvements!

## License

This project is licensed under the [MIT License](LICENSE). Feel free to use and modify the code as per the license terms.

## Acknowledgments

This code example is based on the official Keras documentation and the Keras Tuner library. Check out the official documentation for more details:

- Keras: [https://keras.io/](https://keras.io/)
- TensorFlow: [https://www.tensorflow.org/](https://www.tensorflow.org/)
- Keras Tuner: [https://keras-team.github.io/keras-tuner/](https://keras-team.github.io/keras-tuner/)

## Contact

If you have any questions or suggestions, feel free to contact @tafousignals via email at tafousignals@gmail.com
