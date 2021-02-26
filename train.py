import os
import time
import argparse
from datetime import datetime

import tensorflow as tf
from tensorboard.plugins.hparams import api

import preprocess
import model

def get_dataset(dataset_path,
                batch_size,
                strategy=None):

    # Load dataset
    dataset = preprocess.load_dataset(dataset_path)

    # Pad for batching
    dataset = dataset.padded_batch(
        batch_size, 
        padded_shapes=(
            [-1, -1],       # mel_specs
            [-1])           # labels
    )

    # Prefetch dataset (cuda streaming)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if strategy is not None:
        dataset = strategy.experimental_distribute_dataset(dataset)

    return dataset


def build_model(hp):

    encoder = model.Encoder(
        num_layers=hp["encoder_num_layers"],
        num_units=hp["encoder_num_units"], 
        dropout=hp["encoder_dropout"],
        dropout_prob=hp["encoder_dropout_prob"],
        layer_norm=hp["encoder_layer_norm"],
        dtype=tf.float32
    )

    decoder = model.Decoder(
        attention_unit_num=hp["decoder_attention_unit_num"],
        vocab_size=hp["decoder_vocab_size"],
        embedding_dim=hp["decoder_embeddimg_dim"],
        gru_unit_num=hp["decoder_gru_unit_num"],
        fc_layer_num=hp["decoder_fc_layer_num"],
        fc_unit_num=hp["decoder_fc_unit_num"],
        attention_type=hp["decoder_attention_type"],
        gru_layer_norm=hp["decoder_gru_layer_norm"],
        gru_dropout=hp["decoder_gru_dropout"],
        gru_dropout_prob=hp["decoder_gru_dropout_prob"],
        fc_activation=hp["decoder_fc_activation"],
        dtype=tf.float32
    )

    return encoder, decoder


def evaluate_step(mel, y_true, encoder, decoder, loss_fn):
    
    # Get shapes
    batch_size = y_true.shape[0]
    y_true_length = y_true.shape[1]

    # Set initial states for decoder
    decoder_inputs = tf.zeros(shape=[batch_size, 1], dtype=tf.float32)
    decoder_hidden_states = decoder.get_initial_hidden_states(batch_size)

    # Compute outputs of encoder
    encoder_outputs = encoder(mel)

    loss = float(0)
    y_pred = tf.TensorArray(size=y_true_length, dtype=tf.float32)
    attention_weights = tf.TensorArray(size=y_true_length, dtype=tf.float32)
    for t in tf.range(y_true_length):
        # Comput output of decoder
        decoder_outputs, decoder_hidden_states, attention_weights_t = decoder(decoder_inputs, decoder_hidden_states, encoder_outputs)

        # Compute loss
        loss += tf.reduce_sum(loss_fn(
            tf.cast(tf.expand_dims(y_true[:, t], axis=1), tf.float32),
            decoder_outputs, 
            from_logits=True
        ))

        # Get argmax value for next step
        decoder_inputs = tf.cast(
            tf.expand_dims(tf.argmax(decoder_outputs, axis=1), axis=1),
            dtype=tf.float32
        )

        y_pred = y_pred.write(t, decoder_inputs)
        attention_weights = attention_weights.write(t, attention_weights_t)  # shape of attention_weights_t : [B, L, 1]
        
    loss /= float(batch_size)

    y_pred = y_pred.stack()
    y_pred = tf.cast(tf.transpose(y_pred, [1, 0, 2]), tf.int32)  # [T, B, 1] -> [B, T, 1]
    y_pred = tf.squeeze(y_pred, axis=2)
    cer = tf.reduce_mean(tf.edit_distance(
        hypothesis=tf.sparse.from_dense(y_pred),
        truth=tf.sparse.from_dense(y_true),
        normalize=True
    ))

    attention_weights = attention_weights.stack()  # shape : [T, B, L, 1]
    attention_weights = tf.transpose(attention_weights, [1, 0, 2, 3])  # [T, B, L, 1] -> [B, T, L, 1]
    
    return loss, cer, attention_weights
    
@tf.function
def train_step(mel, y_true, encoder, decoder, optimizer, loss_fn):

    # Get shapes
    batch_size = y_true.shape[0]
    token_time_length = y_true.shape[1]

    # Set initial states for decoder
    decoder_inputs = tf.zeros(shape=[batch_size, 1], dtype=tf.float32)
    decoder_hidden_states = decoder.get_initial_hidden_states(batch_size)

    with tf.GradientTape() as tape:
        # Compute outputs of encoder
        encoder_outputs = encoder(mel)

        loss = float(0)
        for t in tf.range(token_time_length):
            # Comput output of decoder
            decoder_outputs, decoder_hidden_states, _ = decoder(decoder_inputs, decoder_hidden_states, encoder_outputs)

            # Compute loss
            loss += tf.reduce_sum(loss_fn(
                tf.cast(tf.expand_dims(y_true[:, t], axis=1), tf.float32),
                decoder_outputs, 
                from_logits=True
            ))

            # Get argmax value for next step
            decoder_inputs = tf.cast(
                tf.expand_dims(tf.argmax(decoder_outputs, axis=1), axis=1),
                dtype=tf.float32
            )
            
        loss /= float(batch_size)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    
    return loss
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--chkpt_dir", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_mels", type=int, default=80)
    
    parser.add_argument("--encoder_num_layers", type=int, default=4)
    parser.add_argument("--encoder_num_units", type=int, default=256)
    parser.add_argument("--encoder_dropout", type=bool, default=True)
    parser.add_argument("--encoder_dropout_prob", type=float, default=0.1)
    parser.add_argument("--encoder_layer_norm", type=bool, default=True)

    parser.add_argument("--decoder_attention_unit_num", type=int, default=256)
    parser.add_argument("--decoder_embedding_dim", type=int, default=256)
    parser.add_argument("--decoder_gru_unit_num", type=int, default=256)
    parser.add_argument("--decoder_fc_layer_num", type=int, default=2)
    parser.add_argument("--decoder_fc_unit_num", type=int, default=256)
    parser.add_argument("--decoder_attention_type", type=str, default='Bahdanau')
    parser.add_argument("--decoder_gru_dropout", type=bool, default=True)
    parser.add_argument("--decoder_gru_dropout_prob", type=float, default=0.1)
    parser.add_argument("--decoder_gru_layer_norm", type=bool, default=True)
    parser.add_argument("--decoder_fc_activation", type=str, default='relu')
    
    parser.add_argument("--decay_learning_rate", type=bool, default=True)
    parser.add_argument("--init_learning_rate", type=float, default=0.01)
    parser.add_argument("--learning_rate_decay_steps", type=int, default=1000)
    parser.add_argument("--learning_rate_decay_rate", type=float, default=0.96)
    parser.add_argument("--optimizer", type=str, default='sgd', help='one of {"sgd", "adam"}')
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()

    vocab_size = len("abcdefghijklmnopqrstuvwxyz'") + 1
    
    # Set log directory
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    log_dir = os.path.join(args.log_dir, current_time)

    # Set checkpoint directory
    os.makedirs(args.chkpt_dir, exist_ok=True)
    chkpt_dir = os.path.join(args.chkpt_dir, current_time)


    hp = {
        "batch_size": args.batch_size,
        "num_mels": args.num_mels,

        "encoder_num_layers" : args.encoder_num_layers,
        "encoder_num_units" : args.encoder_num_units,
        "encoder_dropout" : args.encoder_dropout,
        "encoder_dropout_prob" : args.encoder_dropout_prob,
        "encoder_layer_norm" : args.encoder_layer_norm,

        "decoder_attention_unit_num": args.decoder_attention_unit_num,
        "decoder_vocab_size": vocab_size,
        "decoder_embeddimg_dim": args.decoder_embedding_dim,
        "decoder_gru_unit_num": args.decoder_gru_unit_num,
        "decoder_fc_layer_num": args.decoder_fc_layer_num,
        "decoder_fc_unit_num": args.decoder_fc_unit_num,
        "decoder_attention_type": args.decoder_attention_type,
        "decoder_gru_dropout": args.decoder_gru_dropout,
        "decoder_gru_dropout_prob": args.decoder_gru_dropout_prob,
        "decoder_gru_layer_norm": args.decoder_gru_layer_norm,
        "decoder_fc_activation": args.decoder_fc_activation,

        "decay_learning_rate" : args.decay_learning_rate,
        "init_learning_rate" : args.init_learning_rate,
        "learning_rate_decay_steps" : args.learning_rate_decay_steps,
        "learning_rate_decay_rate" : args.learning_rate_decay_rate,
        "optimizer" : args.optimizer
    }

    # Get dataset
    train_dataset_path = os.path.join(args.data_dir, "dev.tfrecord")
    dev_dataset_path = os.path.join(args.data_dir, "dev.tfrecord")

    train_dataset = get_dataset(train_dataset_path, hp["batch_size"])
    dev_dataset = get_dataset(dev_dataset_path, hp["batch_size"])

    # Build model
    encoder, decoder = build_model(hp)

    # Build optimizer & loss function
    learning_rate = 0.001
    optimizer = tf.optimizers.Adam(learning_rate)
    loss_fn = tf.losses.sparse_categorical_crossentropy

    # Store hparams
    with tf.summary.create_file_writer(log_dir).as_default():
        api.hparams(hp)

    # Set learning rate
    if not hp["decay_learning_rate"]:
        learning_rate = hp["init_learning_rate"]
    else:
        learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=hp["init_learning_rate"],
            decay_steps=hp["learning_rate_decay_steps"],
            decay_rate=hp["learning_rate_decay_rate"]
        )

    # Set optimizer
    optimizer_type = hp["optimizer"]
    if optimizer_type == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate, momentum=0.9)
    elif optimizer_type == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    else:
        print("undefined optimizer type : {}".format(optimizer_type))
        exit(1)

    global_step = 0
    for epoch in range(args.num_epochs):
        
        # Train
        for batch, (mel, y_true) in enumerate(train_dataset):
            start_time = time.time()

            train_loss = train_step(mel, y_true, encoder, decoder, optimizer, loss_fn)
            step_time = time.time() - start_time

            log_str = 'Epoch : {}, Batch: {}, Global Step : {}, Spent Time : {:.4f}, Loss : {:.4f}'.format(
                epoch, batch, global_step, step_time, train_loss
            )
            print(log_str)

            with tf.summary.create_file_writer(log_dir).as_default():
                tf.summary.scalar("Train Loss", train_loss, step=global_step)

            global_step += 1
            break

        # Evaluate
        loss_object = tf.keras.metrics.Mean()
        cer_object = tf.keras.metrics.Mean()
        for batch, (mel, y_true) in enumerate(dev_dataset.take(1)):
            dev_loss, dev_cer, attention_weights = evaluate_step(mel, y_true, encoder, decoder, loss_fn)
            
            loss_object(dev_loss)
            cer_object(dev_cer)


        dev_loss = loss_object.result()
        dev_cer = cer_object.result()

        log_str = 'Epoch : {}, Batch: {}, Global Step : {}, Loss : {:.4f}, CER : {:.4f}'.format(
            epoch, batch, global_step, dev_loss, dev_cer
        )
        print(log_str)

        with tf.summary.create_file_writer(log_dir).as_default():
            tf.summary.scalar("Dev Loss", dev_loss, step=global_step)
            tf.summary.scalar("Dev cer", dev_cer, step=global_step)
            tf.summary.image("Dev attention", attention_weights, step=global_step)

        # Save checkpoint
        encoder_chkpt_filepath = os.path.join(chkpt_dir, "chkpt_encoder_step:{}_loss:{:.4f}.hdf5".format(global_step, dev_loss))
        decoder_chkpt_filepath = os.path.join(chkpt_dir, "chkpt_decoder_step:{}_loss:{:.4f}.hdf5".format(global_step, dev_loss))
        encoder.save_weights(encoder_chkpt_filepath)
        decoder.save_weights(decoder_chkpt_filepath)