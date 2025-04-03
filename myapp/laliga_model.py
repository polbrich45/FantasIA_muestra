import pandas as pd
import numpy as np
import os
import sys
import random
from collections import deque
import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras import mixed_precision
import matplotlib.pyplot as plt
from IPython.display import clear_output

mixed_precision.set_global_policy('mixed_float16')


def es_relleno(state):
    return np.all(state == 0)

def store_experiences(replay_buffer, state, action, reward, next_state, done):
    if not es_relleno(state) and next_state is not None and not es_relleno(next_state):
        replay_buffer.append((state, action, reward, next_state, done))

def calcular_recompensa(action, reward):
    if action == 0:
        reward = -reward
    return reward

# Capa DuelingLayer para el modelo Dueling DQN
class DuelingLayer(layers.Layer):
    def call(self, inputs):
        value, advantage = inputs
        return value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))

# Crear modelo Dueling DQN con regularización y dropout
def crear_dueling_modelo(n_outputs, optimizer, loss_fn):
    input_layer = layers.Input(shape=(48,))

    # Reducimos el número de neuronas y aplicamos regularización L2 y dropout
    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(input_layer)
    x = layers.Dropout(0.3)(x)  # Dropout del 30%

    x = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.3)(x)  # Dropout del 30%

    # Rama de valor
    value = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    value = layers.Dense(1, activation='linear')(value)

    # Rama de ventaja
    advantage = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    advantage = layers.Dense(n_outputs, activation='linear')(advantage)

    # Capa dueling
    q_values = DuelingLayer()([value, advantage])

    model = Model(inputs=input_layer, outputs=q_values)
    model.compile(optimizer=optimizer, loss=loss_fn)

    return model

# Función para entrenar el modelo usando el replay buffer en GPU
def train_from_replay_buffer(batch_size, replay_buffer, model, target_model, discount_factor):
    if len(replay_buffer) < batch_size:
        return

    mini_batch = random.sample(replay_buffer, batch_size)
    states = np.array([experience[0] for experience in mini_batch])
    actions = np.array([experience[1] for experience in mini_batch])
    rewards = np.array([experience[2] for experience in mini_batch])
    next_states = np.array([experience[3] for experience in mini_batch])
    dones = np.array([experience[4] for experience in mini_batch])

    # Convertir a tensores de GPU
    states = tf.convert_to_tensor(states.reshape(batch_size, 48), dtype=tf.float32)
    next_states = tf.convert_to_tensor(next_states.reshape(batch_size, 48), dtype=tf.float32)

    # Ejecutar predicción en GPU
    with tf.device('/GPU:0'):
        q_values = model.predict(states, verbose=0)
        q_values_next = target_model.predict(next_states, verbose=0)

    for i in range(batch_size):
        if dones[i]:
            q_values[i, actions[i]] = rewards[i]
        else:
            q_values[i, actions[i]] = rewards[i] + discount_factor * np.max(q_values_next[i])

    # Entrenamiento en GPU
    with tf.device('/GPU:0'):
        model.train_on_batch(states, q_values)

# Función principal para entrenar el modelo en GPU
def train_model(X_train, y_train, episodes, n_outputs, optimizer, loss_fn, batch_size, sync_steps, replay_buffer, discount_factor, checkpoint_dir, log_file_path, save_path=None, model=None):
    rewards = []
    accumulated_rewards = []

    # Parámetros de la política ε-greedy
    epsilon = 0.3  # Valor inicial más bajo para más explotación
    epsilon_min = 0.05  # Valor mínimo para permitir aún más explotación al final
    epsilon_decay = 0.999

    # Si no se pasa un modelo, crear uno desde cero
    if model is None:
        print("No se pasó ningún modelo cargado. Creando un nuevo modelo.")
        with tf.device('/GPU:0'):
            model = crear_dueling_modelo(n_outputs, optimizer, loss_fn)
            target_model = crear_dueling_modelo(n_outputs, optimizer, loss_fn)
            target_model.set_weights(model.get_weights())
    else:
        # Usar el modelo cargado
        print("Modelo cargado utilizado para continuar el entrenamiento.")
        target_model = crear_dueling_modelo(n_outputs, optimizer, loss_fn)
        target_model.set_weights(model.get_weights())

    step_count = 0

    for episode in range(episodes):
        episode_reward = 0
        state_index = 0

        while state_index < len(X_train):
            state_sequence = X_train[state_index]

            for partido_index in range(len(state_sequence)):
                state = tf.reshape(tf.convert_to_tensor(state_sequence[partido_index], dtype=tf.float32), (1, -1))

                if es_relleno(state):
                    continue

                # Selección de la acción con ε-greedy
                if np.random.rand() <= epsilon:
                    # Exploración: elegir una acción aleatoria
                    action = np.random.randint(0, n_outputs)
                else:
                    # Explotación: elegir la mejor acción basada en las Q-values
                    q_values = model.predict(state, verbose=0)
                    action = np.argmax(q_values[0])

                target = y_train[state_index][partido_index]
                reward = calcular_recompensa(action, target)

                if reward is None:
                    continue

                next_partido_index = partido_index + 1
                next_state = (tf.reshape(tf.convert_to_tensor(state_sequence[partido_index], dtype=tf.float32), (1, -1))
                            if next_partido_index < len(state_sequence) else None)

                if next_state is not None and not es_relleno(next_state):
                    store_experiences(replay_buffer, state, action, reward, next_state, done=False)

                episode_reward += reward

                # Entrenar desde el replay buffer en GPU
                if len(replay_buffer) > batch_size:
                    train_from_replay_buffer(batch_size, replay_buffer, model, target_model, discount_factor)

                step_count += 1

                # Disminuir epsilon para explorar menos con el tiempo
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay

                # Sincronizar los pesos del target model cada sync_steps
                if step_count % sync_steps == 0:
                    target_model.set_weights(model.get_weights())

                if state_index % 100 == 0:
                    accumulated_reward_current = sum(rewards) + episode_reward
                    with open(log_file_path, 'a') as f:
                        f.write(f"Step: {state_index*(episode+1)}, Episode: {episode + 5}, Total Reward: {episode_reward}, Accumulated Reward: {accumulated_reward_current}\n")
                    print(f"Log saved at step {state_index*(episode+1)}")
                if state_index % 200 == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode + 5}_{state_index}.keras")
                    model.save(checkpoint_path, include_optimizer=True)

                sys.stdout.write(f"\rEpisode: {episode + 1}/{episodes}, State index: {state_index}, Accumulated reward: {episode_reward}")
                sys.stdout.flush()

            state_index += 1

        rewards.append(episode_reward)
        accumulated_rewards.append(sum(rewards))

        # Guardar el modelo al final de cada episodio
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_episode_{episode + 5}.keras")
        model.save(checkpoint_path, include_optimizer=True)

    if save_path:
        model.save(save_path)

    return rewards, accumulated_rewards
