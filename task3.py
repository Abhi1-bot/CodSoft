import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Embedding, Dense, Add, Input
import numpy as np

base_model = ResNet50(weights='imagenet')
feature_extractor = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

def extract_features(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return feature_extractor.predict(img).reshape(1, -1)

captions_dict = {"image1.jpg": "a dog playing with a ball", "image2.jpg": "a cat sitting on a sofa"}
all_captions = list(captions_dict.values())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(c.split()) for c in all_captions)
sequences = tokenizer.texts_to_sequences(all_captions)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

embedding_dim = 256
image_input = Input(shape=(2048,))
image_features = Dense(embedding_dim, activation='relu')(image_input)

text_input = Input(shape=(max_length,))
text_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(text_input)
text_lstm = LSTM(256, return_sequences=True)(text_embedding)
text_lstm = LSTM(256)(text_lstm)

merged_features = Add()([image_features, text_lstm])
output_layer = Dense(vocab_size, activation='softmax')(merged_features)

model = Model(inputs=[image_input, text_input], outputs=output_layer)
model.compile(loss='categorical_crossentropy', optimizer='adam')

X_images = np.random.rand(len(all_captions), 2048)
X_texts = np.random.randint(1, vocab_size, size=(len(all_captions), max_length))
y_labels = np.random.rand(len(all_captions), vocab_size)

model.fit([X_images, X_texts], y_labels, epochs=5, batch_size=32)

def generate_caption(image_path):
    image_features = extract_features(image_path)
    sequence = [tokenizer.word_index.get('startseq', 1)]
    for _ in range(max_length):
        padded_seq = pad_sequences([sequence], maxlen=max_length, padding='post')
        preds = model.predict([image_features, padded_seq])
        next_word_id = np.argmax(preds)
        if next_word_id == tokenizer.word_index.get('endseq', 2):
            break
        sequence.append(next_word_id)
    caption = ' '.join([tokenizer.index_word.get(i, '') for i in sequence if i in tokenizer.index_word])
    return caption

print(generate_caption("test_image.jpg"))
