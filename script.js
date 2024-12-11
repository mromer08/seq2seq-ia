let encoderModel, decoderModel;

/**
 * Normaliza el texto.
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
 * Convierte una frase a secuencia de índices (INT32).
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
    // Usar int32, ya que normalmente las entradas a embeddings son int32
    return tf.tensor([tokens], [1, max_length_inp], 'int32');
}

/**
 * Genera estado oculto inicial (float32 si el modelo así lo requiere).
 */
function initializeHiddenState() {
    return tf.zeros([1, units], 'float32');
}

/**
 * Cargar modelos.
 */
async function loadModels() {
    encoderModel = await tf.loadGraphModel('./encoder_model_js/model.json');
    decoderModel = await tf.loadGraphModel('./decoder_model_js/model.json');
    console.log("Modelos cargados exitosamente!");

    console.log("Entradas encoder:", encoderModel.inputs);
    console.log("Salidas encoder:", encoderModel.outputs);
    console.log("Entradas decoder:", decoderModel.inputs);
    console.log("Salidas decoder:", decoderModel.outputs);
}

/**
 * Evaluar la frase.
 */
async function evaluate(sentence) {
    sentence = preprocessSentence(sentence);
    const inputs = tokenizeInput(sentence);
    let hidden = initializeHiddenState();

    // Llamar sin especificar salidas
    // Usar array en lugar de objeto si el objeto falla.
    // Primero prueba con objeto usando exactamente los nombres dados por encoderModel.inputs.
    // Si no funciona, intenta con array:
    // const encOutputAndState = await encoderModel.executeAsync([inputs, hidden]);
    
    // Si los nombres son exactamente 'keras_tensor_17' y 'keras_tensor_18' sin :0
    const encOutputAndState = await encoderModel.executeAsync({
        "keras_tensor_17": inputs,
        "keras_tensor_18": hidden
    });

    let enc_out = encOutputAndState[0];
    let enc_hidden = encOutputAndState[1];

    let dec_hidden = enc_hidden;
    let dec_input = tf.tensor([[targ_lang_word_index['<start>']]], [1,1], 'int32'); // también int32

    let result = "";
    for (let t = 0; t < max_length_targ; t++) {
        // De nuevo, sin especificar salidas.
        // Igual, si no funciona con objeto, prueba con array:
        // const decOutputAndState = await decoderModel.executeAsync([dec_input, dec_hidden, enc_out]);

        const decOutputAndState = await decoderModel.executeAsync({
            'keras_tensor_21': dec_input,
            'keras_tensor_22': dec_hidden,
            'keras_tensor_23': enc_out
        });

        let predictions = decOutputAndState[0];
        dec_hidden = decOutputAndState[1];

        const predicted_id = (await predictions.argMax(-1).data())[0];
        const predicted_word = targ_lang_index_word[predicted_id];

        if (predicted_word === '<end>') {
            break;
        }

        result += predicted_word + " ";

        dec_input = tf.tensor([[predicted_id]], [1,1], 'int32');
    }

    return result.trim();
}

/**
 * Enviar mensaje.
 */
async function sendMessage() {
    const userInputElem = document.getElementById('userInput');
    const chatContainer = document.getElementById('chatContainer');
    const userText = userInputElem.value.trim();
    if (!userText) return;

    let userMessageDiv = document.createElement('div');
    userMessageDiv.textContent = "Usuario: " + userText;
    userMessageDiv.style.fontWeight = "bold";
    chatContainer.appendChild(userMessageDiv);

    userInputElem.value = "";

    let botResponse = await evaluate(userText);

    let botMessageDiv = document.createElement('div');
    botMessageDiv.textContent = "Bot: " + botResponse;
    chatContainer.appendChild(botMessageDiv);

    chatContainer.scrollTop = chatContainer.scrollHeight;
}

loadModels();
