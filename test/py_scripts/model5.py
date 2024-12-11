import tensorflow as tf
import unicodedata
import re
import numpy as np
import os
import io
import time
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Cargar el dataset de Hugging Face
ds = load_dataset("bertin-project/alpaca-spanish")

# Nos quedamos solo con las primeras 50 líneas del conjunto de entrenamiento
ds = ds["train"].select(range(50))

# Cada registro tiene: instruction | input | output
# Vamos a concatenar instruction + input como entrada y output como salida.
sources = []
targets = []
for example in ds:
    # Concatenamos instruction y input (si input no está vacío, lo añadimos)
    inp_text = example["instruction"]
    if example["input"].strip():
        inp_text += " " + example["input"].strip()
    out_text = example["output"]

    sources.append(inp_text)
    targets.append(out_text)


def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) 
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # Espaciado alrededor de puntuaciones
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^a-zA-Z0-9¿?!.,]+", " ", w)
    w = w.strip()

    # Agregar tokens de inicio y fin
    w = '<start> ' + w + ' <end>'
    return w

# Preprocesamos las oraciones
inp_sentences = [preprocess_sentence(s) for s in sources]
targ_sentences = [preprocess_sentence(t) for t in targets]

# Tokenización
def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer

input_tensor, inp_lang_tokenizer = tokenize(inp_sentences)
target_tensor, targ_lang_tokenizer = tokenize(targ_sentences)

max_length_inp, max_length_targ = input_tensor.shape[1], target_tensor.shape[1]


import json

# Guardar el word_index del input tokenizer
with open("inp_lang_word_index.json", "w", encoding="utf-8") as f:
    json.dump(inp_lang_tokenizer.word_index, f, ensure_ascii=False, indent=4)

# Guardar el word_index del target tokenizer
with open("targ_lang_word_index.json", "w", encoding="utf-8") as f:
    json.dump(targ_lang_tokenizer.word_index, f, ensure_ascii=False, indent=4)

# Opcionalmente, guardar las longitudes máximas y otros parámetros
tokenizer_data = {
    "max_length_inp": max_length_inp,
    "max_length_targ": max_length_targ
}
with open("tokenizer_data.json", "w", encoding="utf-8") as f:
    json.dump(tokenizer_data, f, ensure_ascii=False, indent=4)


# División en entrenamiento y validación
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(
    input_tensor, target_tensor, test_size=0.2
)

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 8  # Más pequeño, ya que son pocas muestras
steps_per_epoch = len(input_tensor_train) // BATCH_SIZE
embedding_dim = 256
units = 512

vocab_inp_size = len(inp_lang_tokenizer.word_index) + 1
vocab_targ_size = len(targ_lang_tokenizer.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state, attention_weights

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_targ_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)
        
        # Teacher forcing
        for t in range(1, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

EPOCHS = 5  # Pocas épocas por ejemplo

for epoch in range(EPOCHS):
    start = time.time()
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 10 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy()))

    print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang_tokenizer.word_index.get(i, 0) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)
    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang_tokenizer.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_word = targ_lang_tokenizer.index_word.get(predicted_id, '')

        if predicted_word == '<end>':
            break

        result += predicted_word + ' '
        dec_input = tf.expand_dims([predicted_id], 0)

    return result.strip()

# Probar el chatbot con alguna entrada
print("Prueba del chatbot:")
user_input = "¿Cuál es la capital de Francia?"
response = evaluate(user_input)
print("Usuario: ", user_input)
print("Chatbot: ", response)


# Creamos un modelo simple para exportar, por ejemplo el encoder
# Para que la exportación sea más sencilla, definamos un modelo Keras funcional que contenga el encoder:
encoder_input = tf.keras.Input(shape=(max_length_inp,))
encoder_hidden_input = tf.keras.Input(shape=(units,))
enc_out, enc_state = encoder(encoder_input, encoder_hidden_input)
encoder_model = tf.keras.Model(inputs=[encoder_input, encoder_hidden_input], outputs=[enc_out, enc_state])

# De igual forma el decoder:
decoder_input = tf.keras.Input(shape=(1,))
decoder_state_input = tf.keras.Input(shape=(units,))
decoder_enc_out_input = tf.keras.Input(shape=(max_length_inp, units))
dec_outputs, dec_state, att_weights = decoder(decoder_input, decoder_state_input, decoder_enc_out_input)
decoder_model = tf.keras.Model(inputs=[decoder_input, decoder_state_input, decoder_enc_out_input],
                               outputs=[dec_outputs, dec_state, att_weights])

# Guardar ambos modelos en formato SavedModel
encoder_model.export("encoder_model")
decoder_model.export("decoder_model")

# Ahora podemos convertir estos modelos a TF.js con:
# !tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model ./encoder_model ./encoder_model_js
# !tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model ./decoder_model ./decoder_model_js


# !tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model ./encoder_model ./encoder_model_js
# !tensorflowjs_converter --input_format=tf_saved_model --output_format=tfjs_graph_model ./decoder_model ./decoder_model_js
