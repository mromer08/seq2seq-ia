import torch
from tokenizers import Tokenizer
from chatbot import GPTLanguageModel

# Configuración
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar el tokenizador y el modelo
tokenizer = Tokenizer.from_file("tokenizer-bpe.json")

vocab_size = tokenizer.get_vocab_size()
model = GPTLanguageModel(vocab_size).to(device)
model.load_state_dict(torch.load("chatbot_model.pth"))
model.eval()  # Establecer el modelo en modo evaluación

# Función principal para el chatbot
while True:
    prompt = input("Usuario: ")
    if prompt.lower() == "salir":
        break

    # Codificar la entrada del usuario
    context = torch.tensor([tokenizer.encode(prompt).ids], device=device)
    
    # Generar la respuesta con el modelo
    with torch.no_grad():
        response_ids = model.generate(context, max_new_tokens=100)
    
    # Decodificar y mostrar la respuesta
    response = tokenizer.decode(response_ids[0].tolist())
    print(f"Chatbot: {response}")
