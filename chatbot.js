// No uses import, ya que TensorFlow.js ya está cargado en index.html
// tf ya estará disponible globalmente

// Cargar el modelo
const loadModel = async () => {
    const model = await tf.loadLayersModel('./model_js/model.json');
    return model;
};

const loadTokenizer = async (path) => {
    const response = await fetch(path);
    const wordIndex = await response.json();
    return wordIndex;
};

// Cargar los tokenizers
(async () => {
    const inputTokenizer = await loadTokenizer('./input_tokenizer.json');
    const outputTokenizer = await loadTokenizer('./output_tokenizer.json');

    const textToSequence = (text, tokenizer) => {
        const words = text.split(' ');
        return words.map((word) => tokenizer[word] || 0); // Si no existe, asignar 0 (padding)
    };

    const padSequence = (sequence, maxLength) => {
        while (sequence.length < maxLength) {
            sequence.push(0); // Añadir padding
        }
        return sequence.slice(0, maxLength); // Limitar al tamaño máximo
    };

    const maxInputLength = 10; // Usa el mismo tamaño definido en Python
    const inputText = "hola";
    const inputSequence = padSequence(textToSequence(inputText, inputTokenizer), maxInputLength);

    const model = await loadModel();
    const response = await generateResponse(inputSequence, model, outputTokenizer);

    console.log("Respuesta del chatbot:", response);

    async function generateResponse(inputSequence, model, outputTokenizer, maxOutputLength = 10) {
        let states = await model.predict(tf.tensor([inputSequence])); // Salida del encoder
        let response = [];
        let currentWord = "<sos>"; // Comenzar con la etiqueta de inicio

        for (let i = 0; i < maxOutputLength; i++) {
            const token = outputTokenizer[currentWord] || 0;
            const prediction = await model.predict([tf.tensor([[token]]), states]);

            // Obtener la palabra con la mayor probabilidad
            const predictedIndex = prediction.argMax(-1).dataSync()[0];
            currentWord = Object.keys(outputTokenizer).find(key => outputTokenizer[key] === predictedIndex);

            if (currentWord === "<eos>" || !currentWord) break; // Detener si llega a <eos>
            response.push(currentWord);
        }

        return response.join(' ');
    }
})();
