
/**
 * Preprocesa la oración de entrada.
 * @param {string} w - La oración de entrada.
 * @returns {string} - La oración preprocesada.
 */
function preprocessSentence(w) {
    w = w.toLowerCase().trim();
    w = w.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
    w = w.replace(/([?.!,¿])/g, " $1 ");
    w = w.replace(/\s+/g, " ");
    w = w.trim();
    w = "<start> " + w + " <end>";
    return w;
}

/**
 * Tokeniza la oración de entrada.
 * @param {string} sentence - La oración preprocesada.
 * @returns {tf.Tensor} - El tensor de entrada tokenizado.
 */
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

/**
 * Inicializa el estado oculto.
 * @returns {tf.Tensor} - El tensor de estado oculto inicial.
 */
function initializeHiddenState() {
    return tf.zeros([1, units], 'float32');
}

/**
 * Carga los modelos de encoder y decoder.
 */
async function loadModels() {
    try {
        // Cambia a loadLayersModel si tus modelos son de Keras
        encoderModel = await tf.loadLayersModel('./encoder_model_js/model.json');
        decoderModel = await tf.loadLayersModel('./decoder_model_js/model.json');
        console.log("Modelos cargados exitosamente!");

        console.log("Entradas encoder:", encoderModel.inputs);
        console.log("Salidas encoder:", encoderModel.outputs);
        console.log("Entradas decoder:", decoderModel.inputs);
        console.log("Salidas decoder:", decoderModel.outputs);
    } catch (error) {
        console.error("Error al cargar los modelos:", error);
    }
}

/**
 * Evalúa una oración y genera una respuesta utilizando el modelo encoder-decoder.
 * @param {string} sentence - La oración de entrada del usuario.
 * @returns {string} - La respuesta generada por el bot.
 */
async function evaluate(sentence) {
    // Preprocesar y tokenizar la oración de entrada
    sentence = preprocessSentence(sentence);
    const inputs = tokenizeInput(sentence);
    let hidden = initializeHiddenState();

    let result = "";

    try {
        // Ejecutar el encoder
        // Asumimos que encoderModel.predict devuelve [enc_out, enc_hidden]
        const encOutputs = encoderModel.predict([inputs, hidden]);
        const enc_out = encOutputs[0];
        const enc_hidden = encOutputs[1];

        // Liberar tensores iniciales
        inputs.dispose();
        hidden.dispose();

        let dec_hidden = enc_hidden;
        let dec_input = tf.tensor([[targ_lang_word_index['<start>']]], [1, 1], 'float32');

        for (let t = 0; t < max_length_targ; t++) {
            // Ejecutar el decoder
            // Asumimos que decoderModel.predict devuelve [predictions, dec_hidden]
            const decOutputs = decoderModel.predict([dec_input, dec_hidden, enc_out]);
            const predictions = decOutputs[0];
            dec_hidden = decOutputs[1];

            // Obtener el id de la palabra predicha
            const predicted_id = (await predictions.argMax(-1).data())[0];
            const predicted_word = targ_lang_index_word[predicted_id];

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

/**
 * Envía un mensaje y recibe una respuesta del bot.
 */
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
