let encoderModel, decoderModel;

function preprocessSentence(w) {
    w = w.toLowerCase().trim();
    w = w.normalize("NFD").replace(/[\u0300-\u036f]/g, "");
    w = w.replace(/([?.!,¿])/g, " $1 ");
    w = w.replace(/\s+/g, " ");
    w = w.trim();
    w = "<start> " + w + " <end>";
    return w;
}

function tokenizeInput(sentence) {
    const words = sentence.split(" ");
    let tokens = words.map(word => inp_lang_word_index[word] || 0);
    while (tokens.length < max_length_inp) {
        tokens.push(0);
    }
    if (tokens.length > max_length_inp) {
        tokens = tokens.slice(0, max_length_inp);
    }
    // Ajusta a int32 si tu embedding original era int32:
    return tf.tensor([tokens], [1, max_length_inp], 'int32');
}

function initializeHiddenState() {
    // El hidden suele ser float32. Ajusta si era int32 en tu entrenamiento.
    return tf.zeros([1, units], 'float32');
}

async function loadModels() {
    encoderModel = await tf.loadGraphModel('./encoder_model_js/model.json');
    decoderModel = await tf.loadGraphModel('./decoder_model_js/model.json');
    console.log("Modelos cargados exitosamente!");

    console.log("Entradas encoder:", encoderModel.inputs);
    console.log("Salidas encoder:", encoderModel.outputs);
    console.log("Entradas decoder:", decoderModel.inputs);
    console.log("Salidas decoder:", decoderModel.outputs);
}

async function evaluate(sentence) {
    sentence = preprocessSentence(sentence);
    const inputs = tokenizeInput(sentence);
    let hidden = initializeHiddenState();

    // Usa model.execute() en lugar de executeAsync().
    // Pasa las entradas como array en el orden definido por encoderModel.inputs:
    // encoderModel.inputs[0] -> keras_tensor_17 (tus inputs)
    // encoderModel.inputs[1] -> keras_tensor_18 (tu hidden)
    const encOutputAndState = encoderModel.execute([inputs, hidden]);

    let enc_out = encOutputAndState[0];
    let enc_hidden = encOutputAndState[1];

    let dec_hidden = enc_hidden;
    let dec_input = tf.tensor([[targ_lang_word_index['<start>']]], [1,1], 'int32');

    let result = "";
    for (let t = 0; t < max_length_targ; t++) {
        // Para el decoder:
        // decoderModel.inputs[0] -> keras_tensor_21 (dec_input)
        // decoderModel.inputs[1] -> keras_tensor_22 (dec_hidden)
        // decoderModel.inputs[2] -> keras_tensor_23 (enc_out)
        const decOutputAndState = decoderModel.execute([dec_input, dec_hidden, enc_out]);
        
        let predictions = decOutputAndState[0];
        dec_hidden = decOutputAndState[1];

        const predicted_id = predictions.argMax(-1).dataSync()[0];
        const predicted_word = targ_lang_index_word[predicted_id];

        if (predicted_word === '<end>') {
            break;
        }

        result += predicted_word + " ";

        dec_input = tf.tensor([[predicted_id]], [1,1], 'int32');
    }

    return result.trim();
}

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
