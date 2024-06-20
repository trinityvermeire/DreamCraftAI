import numpy as np
import tensorflow as tf

SAVE_LOCATION = './audio_model' # Where the models will be saved to and loaded from

START_GRADIENT = 60 # Amount of minutes until the alarm in which the reward function starts changing
END_GRADIENT = 10   # Amount of minutes until the alarm in which the reward function stops changing

input_shape = (4,)   # How many elements will be in the input list
# Current EEG frequency
# Current Sleep Stage
# Predicted Sleep Stage
# Minutes Until Alarm
output_shape = (10,)  # Single continuous output for frequency value
actions = ['0.9_Hz_Delta',
           '3_Hz_Delta',
           '4_Hz_Delta_Isochronic_Pulses',
           '6_Hz_Theta_Isochronic_Pulses',
           '7_Hz_Theta',
           '10_Hz__Alpha_Isochronic_Pulses',
           '12_Hz_Alpha',
           '20_Hz_Beta',
           '40_Hz_Gamma',
           '40_Hz_Gamma_Isochronic_Pulses']  # Array of output options

def gradient(time_until_alarm, startValue, endValue): # Returns appropriate reward float value based on the parameters
    if time_until_alarm > START_GRADIENT:   # Before gradient starts
        return startValue
    elif time_until_alarm > END_GRADIENT:   # Gradient function (linear)
        return (startValue-endValue)*(time_until_alarm-END_GRADIENT)/(START_GRADIENT-END_GRADIENT)+endValue
    else:                                   # After gradient ends
        return endValue

def calculate_reward(predicted_sleep_stage, time_until_alarm):
    if predicted_sleep_stage == 'Sleep stage 4':
        return gradient(time_until_alarm, 1.0, 0.0)
    elif predicted_sleep_stage == 'Sleep stage 3':
        return gradient(time_until_alarm, 0.8, 0.0)
    elif predicted_sleep_stage == 'Sleep stage 2':
        return gradient(time_until_alarm, 0.5, 0.2)
    elif predicted_sleep_stage == 'Sleep stage 1':
        return gradient(time_until_alarm, 0.2, 0.7)
    elif predicted_sleep_stage == 'Sleep stage R':
        return 1.0
    elif predicted_sleep_stage == 'Sleep stage W':
        return gradient(time_until_alarm, 0.0, 1.0)
    elif predicted_sleep_stage == 'Movement time':
        return gradient(time_until_alarm, 0.5, 1.0)
    else:
        raise ValueError("Unrecognized sleep stage: {}".format(predicted_sleep_stage))

'''
Uses the Reccurent Neural Network(RNN) architecture.
It initializes with specified input and output shapes and builds a neural network model with two hidden layers 
of 64 neurons each using ReLU activation functions. The output layer produces linear predictions. 
The model can be called with inputs to predict actions and updated with rewards during training.
It provides functionality to save and load model parameters from h5 files.
'''
class SleepOptomizer1(tf.keras.Model):
    def __init__(self, num_features, num_actions, hidden_units=64, learning_rate=0.001):
        super(SleepOptomizer2, self).__init__()
        self.num_features = num_features
        self.num_actions = num_actions
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate

        # Define layers
        self.rnn_layer = tf.keras.layers.SimpleRNN(self.hidden_units, activation='relu')
        self.hidden_layer = tf.keras.layers.Dense(self.hidden_units, activation='relu')
        self.output_layer = tf.keras.layers.Dense(self.num_actions, activation='softmax')

        # Define loss function and optimizer
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def call(self, inputs):
        with tf.GradientTape() as tape:
            output = self.output_layer(self.hidden_layer(self.rnn_layer(inputs)))
            
            # Sample action from the output distribution
            action_indices = tf.random.categorical(output, 1)[:, 0]
            action = tf.cast(action_indices, dtype=tf.int32)
            
            # Compute one-hot encoded targets
            targets = tf.one_hot(action_indices, depth=self.num_actions)
            
            # Compute loss
            loss = self.loss_fn(targets, output)
            
            # Compute gradients and apply updates
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return action, loss

    def save(self):
        self.save_weights(SAVE_LOCATION+".h5")

    def load(self):
        self.load_weights(SAVE_LOCATION+".h5")
        self.rnn_layer.reset_states() # Resets the reccurent nature of it so the model starts fresh


# Create the model
#model = SleepOptimizer1(input_shape, output_shape)

# Train the model:
#model.call(Current EEG frequency, Predicted EEG frequency, Predicted Sleep Stage, Minutes Until Alarm)

# Save the trained model
#model.save()

# Load the model from file
#model.load()

'''
Policy-Based Optimization Model
It initializes with the number of actions and features, along with learning rate and discount factor parameters. 
The model selects actions based on probabilities and updates policy parameters using rewards.
Uses Xavier initialization(also known as Glorot initialization) as a scaling factor to assign random values to the policy_parameters.
It provides functionality to save and load model parameters from text files.

Learning Rate: This parameter controls the step size or rate at which the model parameters are updated during training.
    A smaller learning rate results in slower but more stable learning, while a larger learning rate can lead to faster
    convergence but may cause instability or oscillations in training. Adjusting the learning rate involves finding a
    balance between convergence speed and stability.

Discount Factor: This parameter determines the importance of future rewards compared to immediate rewards. It discounts
    the value of future rewards in calculations of the expected cumulative reward (often referred to as the "discounted return").
    A discount factor close to 1 assigns more weight to future rewards, encouraging the model to prioritize long-term rewards.
    Conversely, a discount factor closer to 0 gives less weight to future rewards, leading to a more myopic decision-making approach.
    The choice of discount factor depends on the specific task and the importance of long-term versus short-term rewards.
'''
class SleepOptimizer2:
    def __init__(self, num_features, num_actions, learning_rate=0.01, discount_factor=0.99):
        self.num_features = num_features
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        scale = np.sqrt(2 / (num_features + num_actions))  # Scaling factor for Xavier initialization
        self.policy_parameters = np.random.normal(loc=0, scale=scale, size=(num_features, num_actions))

    def call(self, inputs): # returns output and trains the model
        logits = np.dot(inputs, self.policy_parameters)
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        action_index = np.random.choice(range(self.num_actions), p=probabilities)
        action = actions[action_index]  # Select action based on index
        gradient = np.outer(inputs, probabilities - (action_index == np.arange(self.num_actions)))
        reward = calculate_reward(inputs[-2], inputs[-1])
        self.policy_parameters += self.learning_rate * gradient * reward
        return action

    def save(self):
        # Save string to text file
        with open(SAVE_LOCATION+'.txt', 'w') as f:
            f.write('\n'.join(' '.join(map(str, row)) for row in self.policy_parameters))

    def load(self):
        # Load string from text file and load it into the policy_parameters
        with open(SAVE_LOCATION+'.txt', 'r') as f:
            self.policy_parameters = np.array([[float(num) for num in row.split()] for row in f.read().split('\n')])


# Create the model
#model = SleepOptimizer2(input_shape, output_shape)

# Train the model:
#model.call(Current EEG frequency, Predicted EEG frequency, Predicted Sleep Stage, Minutes Until Alarm)

# Save the trained model
#model.save()

# Load the model from file
#model.load()