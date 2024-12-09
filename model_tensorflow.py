import tensorflow as tf
from tensorflow.keras import layers, Model
from tokenizers import Tokenizer
import numpy as np

# Configuración del dispositivo
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

# Cargar el tokenizador entrenado previamente
tokenizer = Tokenizer.from_file("tokenizer-bpe.json")
vocab_size = tokenizer.get_vocab_size()

# Parámetros del modelo
block_size = 128
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
batch_size = 64
learning_rate = 3e-4
num_epochs = 1

# Dataset personalizado
class ChatDataset:
    def __init__(self, data_path, tokenizer, block_size):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = f.read()
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.encoded_data = tokenizer.encode(self.data).ids

    def __len__(self):
        return len(self.encoded_data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.encoded_data[idx:idx + self.block_size + 1]
        return (np.array(chunk[:-1], dtype=np.int32), 
                np.array(chunk[1:], dtype=np.int32))

# Cargar datos
data_path = "es.txt"
dataset = ChatDataset(data_path, tokenizer, block_size)
def data_generator():
    for i in range(len(dataset)):
        yield dataset[i]

train_data = tf.data.Dataset.from_generator(data_generator, 
                                            output_signature=(
                                                tf.TensorSpec(shape=(block_size,), dtype=tf.int32),
                                                tf.TensorSpec(shape=(block_size,), dtype=tf.int32)))
train_data = train_data.batch(batch_size).shuffle(1000)

# Bloques del modelo
class MultiHeadAttention(layers.Layer):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.key = layers.Dense(num_heads * head_size, use_bias=False)
        self.query = layers.Dense(num_heads * head_size, use_bias=False)
        self.value = layers.Dense(num_heads * head_size, use_bias=False)
        self.dropout = layers.Dropout(dropout)
        self.proj = layers.Dense(n_embd)

    def call(self, x):
        B, T, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        k = tf.reshape(self.key(x), (B, T, self.num_heads, self.head_size))
        q = tf.reshape(self.query(x), (B, T, self.num_heads, self.head_size))
        v = tf.reshape(self.value(x), (B, T, self.num_heads, self.head_size))

        k = tf.transpose(k, [0, 2, 1, 3])
        q = tf.transpose(q, [0, 2, 1, 3])
        v = tf.transpose(v, [0, 2, 1, 3])

        scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_size, tf.float32))
        mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)  # Matriz triangular inferior
        scores = tf.where(mask == 0, -1e9, scores)
        attention_weights = tf.nn.softmax(scores, axis=-1)
        attention_weights = self.dropout(attention_weights)

        out = tf.matmul(attention_weights, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, (B, T, self.num_heads * self.head_size))
        return self.proj(out)

class FeedForward(layers.Layer):
    def __init__(self, n_embd):
        super().__init__()
        self.net = tf.keras.Sequential([
            layers.Dense(4 * n_embd, activation="relu"),
            layers.Dense(n_embd),
            layers.Dropout(dropout)
        ])

    def call(self, x):
        return self.net(x)

class Block(layers.Layer):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def call(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class GPTLanguageModel(Model):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table = layers.Embedding(block_size, n_embd)
        self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = layers.LayerNormalization()
        self.lm_head = layers.Dense(vocab_size)

    def call(self, index, targets=None):
        B, T = tf.shape(index)[0], tf.shape(index)[1]
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(tf.range(T))
        x = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        else:
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets))
            return logits, loss

    def generate(self, index, max_new_tokens):
        for _ in range(max_new_tokens):
            index_cond = index[:, -block_size:]
            logits, _ = self.call(index_cond)
            logits = logits[:, -1, :]
            probs = tf.nn.softmax(logits, axis=-1)
            index_next = tf.random.categorical(probs, num_samples=1)
            index = tf.concat([index, index_next], axis=1)
        return index

# Inicializar el modelo
with tf.device(device):
    model = GPTLanguageModel(vocab_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (x, y) in enumerate(train_data):
            with tf.GradientTape() as tape:
                logits, loss = model(x, y)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            total_loss += loss.numpy()
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.numpy():.4f}")

        avg_loss = total_loss / len(train_data)
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")

    # Guardar el modelo entrenado
    model.save_weights("chatbot_model.h5")
    print("Modelo entrenado y guardado como 'chatbot_model.h5'")
