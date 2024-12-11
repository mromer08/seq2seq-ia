let encoderModel, decoderModel;

// Función para preprocesar la oración de entrada
function preprocessSentence(w) {
    w = w.toLowerCase().trim();
    w = w.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
    w = w.replace(/([?.!,¿])/g, " $1 ");
    w = w.replace(/\s+/g, " ");
    w = w.trim();
    w = "<start> " + w + " <end>";
    return w;
}

// Función para tokenizar la entrada
function tokenizeInput(sentence) {
    const words = sentence.split(" ");
    let tokens = words.map(word => inp_lang_word_index[word] || 0);
    while (tokens.length < max_length_inp) {
        tokens.push(0);
    }
    if (tokens.length > max_length_inp) {
        tokens = tokens.slice(0, max_length_inp);
    }
    // Ajusta a float32 si tu embedding original era float32:
    return tf.tensor([tokens], [1, max_length_inp], 'float32');
}

// Función para inicializar el estado oculto
function initializeHiddenState() {
    // El hidden suele ser float32. Ajusta si era float32 en tu entrenamiento.
    return tf.zeros([1, units], 'float32');
}

// Función para cargar los modelos de encoder y decoder
async function loadModels() {
    try {
        encoderModel = await tf.loadGraphModel('./encoder_model_js/model.json');
        decoderModel = await tf.loadGraphModel('./decoder_model_js/model.json');
        console.log("Modelos cargados exitosamente!");

        console.log("Entradas encoder:", encoderModel.inputs);
        console.log("Salidas encoder:", encoderModel.outputs);
        console.log("Entradas decoder:", decoderModel.inputs);
        console.log("Salidas decoder:", decoderModel.outputs);
    } catch (error) {
        console.error("Error al cargar los modelos:", error);
    }
}

// Función para evaluar una oración y generar una respuesta
async function evaluate(sentence) {
    sentence = preprocessSentence(sentence);
    const inputs = tokenizeInput(sentence);
    let hidden = initializeHiddenState();

    let result = "";

    try {
        // Depuración: Mostrar las formas de los tensores de entrada
        console.log("Inputs shape:", inputs.shape);
        console.log("Hidden shape:", hidden.shape);

        // Obtener los nombres de las entradas del encoder
        const encoderInputNames = encoderModel.inputs.map(input => input.name);
        console.log("Nombres de entradas del encoder:", encoderInputNames);

        // Preparar las entradas como un objeto con los nombres de las entradas
        const encoderInputs = {};
        encoderModel.inputs.forEach((input, index) => {
            if (index === 0) {
                encoderInputs[input.name] = inputs;
            } else if (index === 1) {
                encoderInputs[input.name] = hidden;
            }
        });

        // Ejecutar el encoder de forma asíncrona
        const encOutputAndState = await encoderModel.executeAsync(encoderInputs);
        console.log("EncOutputAndState:", encOutputAndState);

        // Obtener las salidas del encoder
        const enc_out = encOutputAndState[0];
        const enc_hidden = encOutputAndState[1];

        // Liberar tensores iniciales
        inputs.dispose();
        hidden.dispose();

        let dec_hidden = enc_hidden;
        let dec_input = tf.tensor([[targ_lang_word_index['<start>']]], [1, 1], 'float32');

        for (let t = 0; t < max_length_targ; t++) {
            // Depuración: Mostrar las formas de los tensores de entrada del decoder
            console.log("Decoder Input shape:", dec_input.shape);
            console.log("Decoder Hidden shape:", dec_hidden.shape);
            console.log("Encoder Output shape:", enc_out.shape);

            // Obtener los nombres de las entradas del decoder
            const decoderInputNames = decoderModel.inputs.map(input => input.name);
            console.log("Nombres de entradas del decoder:", decoderInputNames);

            // Preparar las entradas del decoder
            const decoderInputs = {};
            decoderModel.inputs.forEach((input, index) => {
                if (index === 0) {
                    decoderInputs[input.name] = dec_input;
                } else if (index === 1) {
                    decoderInputs[input.name] = dec_hidden;
                } else if (index === 2) {
                    decoderInputs[input.name] = enc_out;
                }
            });

            // Ejecutar el decoder de forma asíncrona
            const decOutputAndState = await decoderModel.executeAsync(decoderInputs);
            console.log("DecOutputAndState:", decOutputAndState);

            const predictions = decOutputAndState[0];
            dec_hidden = decOutputAndState[1];

            // Obtener el id de la palabra predicha
            const predicted_id = predictions.argMax(-1).dataSync()[0];
            const predicted_word = targ_lang_index_word[predicted_id];
            console.log(`Predicted ID: ${predicted_id}, Predicted Word: ${predicted_word}`);

            // Liberar tensores temporales
            predictions.dispose();

            if (predicted_word === '<end>') {
                break;
            }

            result += predicted_word + " ";

            // Liberar el tensor dec_input anterior antes de crear uno nuevo
            dec_input.dispose();
            dec_input = tf.tensor([[predicted_id]], [1, 1], 'float32');
        }

        // Liberar tensores finales
        enc_out.dispose();
        enc_hidden.dispose();
        dec_hidden.dispose();
        dec_input.dispose();

        return result.trim();
    } catch (error) {
        console.error("Error durante la evaluación:", error);
        return "Lo siento, ocurrió un error al procesar tu solicitud.";
    }
}

// Función para enviar un mensaje y recibir una respuesta del bot
async function sendMessage() {
    const userInputElem = document.getElementById('userInput');
    const chatContainer = document.getElementById('chatContainer');
    const userText = userInputElem.value.trim();
    if (!userText) return;

    // Mostrar el mensaje del usuario en el chat
    let userMessageDiv = document.createElement('div');
    userMessageDiv.textContent = "Usuario: " + userText;
    userMessageDiv.style.fontWeight = "bold";
    chatContainer.appendChild(userMessageDiv);

    userInputElem.value = "";

    // Obtener la respuesta del bot
    let botResponse = await evaluate(userText);

    // Mostrar la respuesta del bot en el chat
    let botMessageDiv = document.createElement('div');
    botMessageDiv.textContent = "Bot: " + botResponse;
    chatContainer.appendChild(botMessageDiv);

    // Desplazar el contenedor del chat hacia abajo
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Cargar los modelos al iniciar
loadModels();
