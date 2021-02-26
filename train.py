import os
import argparse

import tensorflow as tf

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
        num_layers=hp["encoder"]["num_layers"],
        num_units=hp["encoder"]["num_units"], 
        dropout=hp["encoder"]["dropout"],
        dropout_prob=hp["encoder"]["dropout_prob"],
        layer_norm=hp["encoder"]["layer_norm"],
        dtype=tf.float32
    )

    decoder = model.Decoder(
        attention_unit_num=hp["decoder"]["attention_unit_num"],
        vocab_size=hp["decoder"]["vocab_size"],
        embedding_dim=hp["decoder"]["embeddimg_dim"],
        gru_unit_num=hp["decoder"]["gru_unit_num"],
        fc_layer_num=hp["decoder"]["fc_layer_num"],
        fc_unit_num=hp["decoder"]["fc_unit_num"],
        attention_type=hp["decoder"]["attention_type"],
        gru_layer_norm=hp["decoder"]["gru_layer_norm"],
        gru_dropout=hp["decoder"]["gru_dropout"],
        gru_dropout_prob=hp["decoder"]["gru_dropout_prob"],
        fc_activation=hp["decoder"]["fc_activation"],
        dtype=tf.float32
    )

    return encoder, decoder


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
    #parser.add_argument("--log_dir", type=str, required=True)
    #parser.add_argument("--chkpt_dir", type=str, required=True)

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
    args = parser.parse_args()

    vocab_size = len("abcdefghijklmnopqrstuvwxyz'") + 1

    hp = {
        "batch_size": args.batch_size,
        "num_mels": args.num_mels,

        "encoder" : {
            "num_layers" : args.encoder_num_layers,
            "num_units" : args.encoder_num_units,
            "dropout" : args.encoder_dropout,
            "dropout_prob" : args.encoder_dropout_prob,
            "layer_norm" : args.encoder_layer_norm,
        },

        "decoder" : {
            "attention_unit_num": args.decoder_attention_unit_num,
            "vocab_size": vocab_size,
            "embeddimg_dim": args.decoder_embedding_dim,
            "gru_unit_num": args.decoder_gru_unit_num,
            "fc_layer_num": args.decoder_fc_layer_num,
            "fc_unit_num": args.decoder_fc_unit_num,
            "attention_type": args.decoder_attention_type,
            "gru_dropout": args.decoder_gru_dropout,
            "gru_dropout_prob": args.decoder_gru_dropout_prob,
            "gru_layer_norm": args.decoder_gru_layer_norm,
            "fc_activation": args.decoder_fc_activation,
        }
    }

    # Get dataset
    dev_dataset_path = os.path.join(args.data_dir, "dev.tfrecord")
    dev_dataset = get_dataset(dev_dataset_path, hp["batch_size"])

    encoder, decoder = build_model(hp)
    learning_rate = 0.001
    optimizer = tf.optimizers.Adam(learning_rate)
    loss_fn = tf.losses.sparse_categorical_crossentropy

    for b, inputs in enumerate(dev_dataset.take(1)):

        mel, y_true = inputs
        
        loss = train_step(mel, y_true, encoder, decoder, optimizer, loss_fn)

        print(loss)