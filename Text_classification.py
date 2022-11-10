#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import file
import pandas as pd
df = pd.read_csv('../content/drive/MyDrive/Colab Notebooks/ecommerceDataset.csv')
df
#rubah judul column
df.rename(columns={'Household': 'Jenis_Produk', 'Paper Plane Design Framed Wall Hanging Motivational Office Decor Art Prints 
                    (8.7 X 8.7 inch) - Set of 4 Painting made up in synthetic frame with uv textured print which gives multi 
                    effects and attracts towards it. This is an special series of paintings which makes your wall very beautiful
                    and gives a royal touch. This painting is ready to hang, you would be proud to possess this unique painting 
                    that is a niche apart. We use only the most modern and efficient printing technology on our prints, 
                    with only the and inks and precision epson, roland and hp printers. This innovative hd printing technique 
                    results in durable and spectacular looking prints of the highest that last a lifetime. We print solely with 
                    top-notch 100% inks, to achieve brilliant and true colours. Due to their high level of uv resistance, 
                    our prints retain their beautiful colours for many years. Add colour and style to your living space with 
                    this digitally printed painting. Some are for pleasure and some for eternal bliss.so bring home this elegant 
                    print that is lushed with rich colors that makes it nothing but sheer elegance to be to your friends and family.
                    it would be treasured forever by whoever your lucky recipient is. Liven up your place with these intriguing 
                    paintings that are high definition hd graphic digital prints for home, office or any room.': 'Keterangan_produk'}, inplace=True)
                    
df.info()

#hapus baris yang bernilai null atau nan
df = df.dropna() 

#mengambil 10% dari dataset 

New_df = df.sample(frac = 0.1)
New_df.shape

#ploting

import matplotlib.pyplot as plt
New_df["Jenis_Produk"].value_counts().plot.bar()

# One-hot encoding

category = pd.get_dummies(New_df['Jenis_Produk'])
df_baru = pd.concat([New_df, category], axis=1)
df_baru = df_baru.drop(columns='Jenis_Produk')
df_baru

#menentukan features dan variabel
text = df_baru['Keterangan_produk']
label = df_baru[['Books', 'Clothing & Accessories', 'Electronics', 'Household']]
#membagi dataset
from sklearn.model_selection import train_test_split
text_latih, text_test, label_latih, label_test = train_test_split(text, label, test_size=0.2, random_state=42)

#tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
 
tokenizer = Tokenizer(num_words=1000, oov_token='x')
tokenizer.fit_on_texts(text_latih) 
tokenizer.fit_on_texts(text_test)
sekuens_latih = tokenizer.texts_to_sequences(text_latih)
sekuens_test = tokenizer.texts_to_sequences(text_test)
padded_latih = pad_sequences(sekuens_latih, maxlen=100, padding = 'post')
padded_test = pad_sequences(sekuens_test,  maxlen=100, padding = 'post')
print (padded_latih.shape)
print (padded_test.shape)

#pemodelan
import tensorflow as tf
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=1000, output_dim=64),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])
model.compile(loss='BinaryCrossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

#callback
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.90):
      print("\nAkurasi telah mencapai >90%!")
      self.model.stop_training = True
callbacks = myCallback()

num_epochs = 100
history = model.fit(padded_latih, label_latih, epochs=num_epochs, callbacks=[callbacks],
                    validation_data=(padded_test, label_test), verbose=2)

#ploting akurasi dan loss
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Akurasi Model')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

