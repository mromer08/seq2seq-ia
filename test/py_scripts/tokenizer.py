# Descargar y descomprimir el dataset en espaniol
# !wget https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/mono/es.txt.gz
# !gunzip es.txt.gz
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

# Configurar el tokenizador
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = Whitespace()

# Configurar el trainer para entrenar el tokenizador
trainer = BpeTrainer(special_tokens=["<pad>", "<sos>", "<eos>"], vocab_size=30000)

# Leer los datos del corpus
paths = ["es.txt"]  # Asegúrate de tener un archivo con diálogos en español
tokenizer.train(files=paths, trainer=trainer)

# Configurar postprocesamiento para usar tokens especiales
tokenizer.post_processor = processors.TemplateProcessing(
    single="<sos> $A <eos>",
    pair="<sos> $A <eos> $B:1 <eos>:1",
    special_tokens=[
        ("<sos>", tokenizer.token_to_id("<sos>")),
        ("<eos>", tokenizer.token_to_id("<eos>")),
    ],
)

# Guardar el tokenizador entrenado
tokenizer.save("tokenizer-bpe.json")
print("Tokenizador entrenado y guardado en 'tokenizer-bpe.json'")
