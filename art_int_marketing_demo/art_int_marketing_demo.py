"""
DOCSTRING
"""
import io
import keras
import numpy
import random
import sys
import tensorrec

class TensorrecExample:

    """
    DOCSTRING
    """

    def __call__(self):
        model = tensorrec.TensorRec()
        interactions, user_features, item_features = tensorrec.util.generate_dummy_data(
            num_users=100, num_items=150, interaction_density=0.05)
        model.fit(interactions, user_features, item_features, epochs=5, verbose=True)
        predictions = model.predict(
            user_features=user_features, item_features=item_features)
        r_at_k = tensorrec.eval.recall_at_k(
            model, interactions, k=10, user_features=user_features, item_features=item_features)
        print(numpy.mean(r_at_k))

class TextGenerationA:

    """
    Sequence to sequence example in Keras (character-level).

    This class demonstrates how to implement a basic character-level sequence-to-sequence model.
    We apply it to translating short English sentences into short French sentences.
    """

    def __init__(self):
        batch_size = 64
        epochs = 50
        latent_dim = 256
        num_samples = 10000
        data_path = 'fra-eng/fra.txt'

    def __call__(self):
        # vectorize the data
        input_texts, target_texts = list(), list()
        input_characters, target_characters = set(), set()
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(num_samples, len(lines) - 1)]:
            input_text, target_text = line.split('\t')
            target_text = '\t' + target_text + '\n'
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)
        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        num_encoder_tokens = len(input_characters)
        num_decoder_tokens = len(target_characters)
        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])
        print('Number of samples:', len(input_texts))
        print('Number of unique input tokens:', num_encoder_tokens)
        print('Number of unique output tokens:', num_decoder_tokens)
        print('Max sequence length for inputs:', max_encoder_seq_length)
        print('Max sequence length for outputs:', max_decoder_seq_length)
        input_token_index = dict(
            [(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict(
            [(char, i) for i, char in enumerate(target_characters)])
        encoder_input_data = numpy.zeros(
            (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
            dtype='float32')
        decoder_input_data = numpy.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
        decoder_target_data = numpy.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
            dtype='float32')
        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                # NOTE: decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, target_token_index[char]] = 1.
                if t > 0:
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.
        # define an input sequence and process it
        encoder_inputs = keras.layers.Input(shape=(None, num_encoder_tokens))
        encoder = keras.layers.LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # discard encoder_outputs and keep the states
        encoder_states = [state_h, state_c]
        # set up the decoder using encoder_state as initial state
        decoder_inputs = keras.layers.Input(shape=(None, num_decoder_tokens))
        decoder_lstm = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(
            decoder_inputs, initial_state=encoder_states)
        decoder_dense = keras.layers.Dense(num_decoder_tokens, activation='softmax')
        decoder_outputs = decoder_dense(decoder_outputs)
        model = keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        # training
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2)
        model.save('s2s.h5')
        # define sampling models
        encoder_model = keras.models.Model(encoder_inputs, encoder_states)
        decoder_state_input_h = keras.layers.Input(shape=(latent_dim,))
        decoder_state_input_c = keras.layers.Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = keras.models.Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        # reverse lookup token index to decode sequences
        reverse_input_char_index = dict(
            (i, char) for char, i in input_token_index.items())
        reverse_target_char_index = dict(
            (i, char) for char, i in target_token_index.items())
        for seq_index in range(100):
            # take one sequence for testing
            input_seq = encoder_input_data[seq_index: seq_index + 1]
            decoded_sentence = decode_sequence(input_seq)
            print('-')
            print('Input sentence:', input_texts[seq_index])
            print('Decoded sentence:', decoded_sentence)

    def decode_sequence(self, input_seq):
        """
        DOCSTRING
        """
        states_value = encoder_model.predict(input_seq)
        target_seq = numpy.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, target_token_index['\t']] = 1.0
        # sampling loop for a batch of sequences
        stop_condition, decoded_sentence = False, ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)
            # sample a token
            sampled_token_index = numpy.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char
            # exit condition: hit max length or find stop character
            if sampled_char == '\n':
                stop_condition = True
            if len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True
            # update the target sequence (of length 1)
            target_seq = numpy.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0
            states_value = [h, c]
        return decoded_sentence

class TextGenerationB:

    """
    Example script to generate text from Nietzsche's writings.

    At least 20 epochs are required before the generated text becomes coherent.
    It is recommended to run this script on a GPU,
    as recurrent networks are quite computationally intensive.
    If you try this script on new data,
    make sure your corpus has at least ~100k characters. ~1M is better.
    """

    def __init__(self):
        path = keras.utils.data_utils.get_file(
            'nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
        with io.open(path, encoding='utf-8') as f:
            text = f.read().lower()
        print('corpus length:', len(text))
        chars = sorted(list(set(text)))
        print('total chars:', len(chars))
        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))
        # cut the text in semi-redundant sequences of maxlen characters
        maxlen, step, sentences, next_chars = 40, 3, list(), list()
        for i in range(len(text) - maxlen, step):
            sentences.append(text[i: i + maxlen])
            next_chars.append(text[i + maxlen])
        print('nb sequences:', len(sentences))
        print('Vectorization...')
        x = numpy.zeros((len(sentences), maxlen, len(chars)), dtype=numpy.bool)
        y = numpy.zeros((len(sentences), len(chars)), dtype=numpy.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

    def __call__(self):
        # build the model - a single LSTM
        print('Build model...')
        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(128, input_shape=(maxlen, len(chars))))
        model.add(keras.layers.Dense(len(chars)))
        model.add(keras.layers.Activation('softmax'))
        optimizer = keras.optimizers.RMSprop(lr=0.01)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        print_callback = keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)
        model.fit(x, y, batch_size=128, epochs=60, callbacks=[print_callback])

    def on_epoch_end(self, epoch, logs):
        """
        Function invoked at end of each epoch. Prints generated text.
        """
        print()
        print('----- Generating text after Epoch: %d' % epoch)
        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)
            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)
            for i in range(400):
                x_pred = numpy.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.
                preds = model.predict(x_pred, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]
                generated += next_char
                sentence = sentence[1:] + next_char
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

    def sample(self, preds, temperature=1.0):
        """
        Helper function to sample an index from a probability array.
        """
        preds = numpy.asarray(preds).astype('float64')
        preds = numpy.log(preds) / temperature
        exp_preds = numpy.exp(preds)
        preds = exp_preds / numpy.sum(exp_preds)
        probas = numpy.random.multinomial(1, preds, 1)
        return numpy.argmax(probas)
