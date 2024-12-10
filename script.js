let encoderModel, decoderModel;

/**
 * Función para normalizar el texto tal como se hizo en Python.
 * Esta es una simplificación. Ajusta las regex según el preprocessing real.
 */
function preprocessSentence(w) {
    w = w.toLowerCase().trim();
    w = w.normalize("NFD").replace(/[\u0300-\u036f]/g, ""); // remover acentos
    w = w.replace(/([?.!,¿])/g, " $1 ");
    w = w.replace(/\s+/g, " ");
    w = w.trim();
    w = "<start> " + w + " <end>";
    return w;
}

/**
 * Convierte una frase en una secuencia de índices usando inp_lang_word_index
 * y la rellena hasta max_length_inp.
 */
function tokenizeInput(sentence) {
    let tokens = sentence.split(" ").map(word => inp_lang_word_index[word] || 0);
    // Rellenar con ceros hasta max_length_inp
    while (tokens.length < max_length_inp) {
        tokens.push(0);
    }
    if (tokens.length > max_length_inp) {
        tokens = tokens.slice(0, max_length_inp);
    }
    return tf.tensor([tokens], [1, max_length_inp], 'int32');
}

/**
 * Genera un vector de estado oculto inicial (cero).
 */
function initializeHiddenState() {
    return tf.zeros([1, units]);
}

/**
 * Función para cargar los modelos TF.js
 */
async function loadModels() {
    encoderModel = await tf.loadGraphModel('./encoder_model_js/model.json');
    decoderModel = await tf.loadGraphModel('./decoder_model_js/model.json');
}

/**
 * Función para evaluar la entrada y obtener la respuesta del chatbot
 */
async function evaluate(sentence) {
    sentence = preprocessSentence(sentence);
    const inputs = tokenizeInput(sentence);

    // Estado inicial del encoder
    let hidden = initializeHiddenState();

    // Ejecutar encoder
    // Encoder model input: [encoder_input, encoder_hidden_input]
    // Ajustar si el modelo exportado tiene nombres distintos de entrada.
    // Suponiendo que el modelo fue guardado con orden establecida:
    const encOutputAndState = encoderModel.execute(
        { 'input_1': inputs, 'input_2': hidden },  // ajustar nombres según tu modelo exportado
        ['Identity','Identity_1'] // ajustar nombres si son distintos (basado en el modelo)
    );
    let enc_out = encOutputAndState[0]; 
    let enc_hidden = encOutputAndState[1];

    // Iniciar decoder
    let dec_hidden = enc_hidden;
    let dec_input = tf.tensor([[targ_lang_word_index['<start>']]], [1,1], 'int32');

    let result = "";
    for (let t = 0; t < max_length_targ; t++) {
        // Decoder input: [decoder_input, decoder_state_input, decoder_enc_out_input]
        const decOutputAndState = decoderModel.execute(
            { 'input_1': dec_input, 'input_2': dec_hidden, 'input_3': enc_out },
            ['Identity','Identity_1','Identity_2'] // ajustar nombres si difieren
        );
        let predictions = decOutputAndState[0];
        dec_hidden = decOutputAndState[1]; 
        // att_weights = decOutputAndState[2] (si lo necesitas)

        // Obtener índice con mayor probabilidad
        const predicted_id = (await predictions.argMax(-1).data())[0];
        const predicted_word = targ_lang_index_word[predicted_id];

        if (predicted_word === '<end>') {
            break;
        }

        result += predicted_word + " ";

        // Actualizar dec_input con la palabra predicha
        dec_input = tf.tensor([[predicted_id]], [1,1], 'int32');
    }

    return result.trim();
}

/**
 * Envía el mensaje del usuario, obtiene la respuesta y la muestra
 */
async function sendMessage() {
    const userInputElem = document.getElementById('userInput');
    const chatContainer = document.getElementById('chatContainer');
    const userText = userInputElem.value.trim();
    if (!userText) return;

    // Mostrar el mensaje del usuario
    let userMessageDiv = document.createElement('div');
    userMessageDiv.textContent = "Usuario: " + userText;
    userMessageDiv.style.fontWeight = "bold";
    chatContainer.appendChild(userMessageDiv);

    // Limpiar el input
    userInputElem.value = "";

    // Obtener respuesta del chatbot
    let botResponse = await evaluate(userText);

    let botMessageDiv = document.createElement('div');
    botMessageDiv.textContent = "Bot: " + botResponse;
    chatContainer.appendChild(botMessageDiv);

    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Cargar los modelos al iniciar
loadModels().then(() => {
    console.log("Modelos cargados exitosamente!");
});
