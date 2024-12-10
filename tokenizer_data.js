// Estos diccionarios son un ejemplo, deben extraerse del entrenamiento real.
window.inp_lang_word_index = {
    "<start>": 1,
    "<end>": 2,
    "¿cuál": 3,
    "es": 4,
    "la": 5,
    "capital": 6,
    "de": 7,
    "francia?": 8,
    // ... y así con todas las palabras de entrada
  };
  
  window.targ_lang_word_index = {
    "<start>": 1,
    "<end>": 2,
    "parís": 3,
    "es": 4,
    "la": 5,
    "capital": 6,
    "de": 7,
    "francia": 8,
    // ... y así con todas las palabras de salida
  };
  
  // Invertimos para obtener index_word
  window.targ_lang_index_word = {};
  for (let word in targ_lang_word_index) {
    targ_lang_index_word[targ_lang_word_index[word]] = word;
  }
  
  // Parámetros del modelo (ejemplo, deben coincidir con el entrenamiento)
  window.max_length_inp = 40;  // ajusta según el entrenamiento real
  window.max_length_targ = 40; // ajusta según el entrenamiento real
  window.units = 512; // ajusta según lo entrenado
  