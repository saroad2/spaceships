from keras import Model, layers

from spaceships.direction import Direction


def get_critic(state_shape):
    state_input = layers.Input(shape=state_shape)
    state_out = layers.Conv2D(16, (3, 3), activation="relu")(state_input)
    state_out = layers.MaxPool2D()(state_out)
    state_out = layers.Conv2D(32, (2, 2), activation="relu")(state_out)
    state_out = layers.Flatten()(state_out)

    action_input = layers.Input(shape=(len(Direction),))
    action_out = layers.Dense(16, activation="relu")(action_input)
    action_out = layers.Dense(32, activation="relu")(action_out)

    concat = layers.Concatenate()([state_out, action_out])
    out = layers.Dense(128, activation="relu")(concat)
    out = layers.Dense(1, activation="relu")(out)

    return Model(inputs=[state_input, action_input], outputs=out)


def update_target(target, model, tau):
    for a_target, a in zip(target.weights, model.weights):
        a_target.assign(tau * a_target + (1 - tau) * a)
