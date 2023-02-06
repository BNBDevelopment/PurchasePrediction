import os
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization
from tensorflow.python.client import device_lib

def buildCustTransactions():
    print("Start...")
    transactions_raw_data = pd.read_csv("./data/transactions_train.csv")
    transactions_raw_data.head()
    transactions = np.array(transactions_raw_data)

    sorted_transactions = transactions[transactions[:, 1].argsort()]
    del transactions
    trs_val, trs_index = np.unique(sorted_transactions[:, 1], return_counts=True)

    cur_idx = 0
    pointer = 0

    all_combined = None
    print("Loop...")
    with open("nn_rich_train.csv", "a") as myfile:
        while pointer < len(trs_index):
            if pointer % 100 == 0:
                print("Loaded: " + str(pointer) + " / " + str(len(trs_index)))

            end_idx = cur_idx + trs_index[pointer]
            all_articles_for_cust = sorted_transactions[cur_idx:end_idx, :]

            times = all_articles_for_cust[:, 0].astype(np.datetime64)
            # firstyr_from_date = datetime(2018, 8, 1) #Aug 01
            # firstyr_to_date = (firstyr_from_date + timedelta(days=120)).date()
            # firstidx = (times > firstyr_from_date.date()) & (times <= firstyr_to_date)
            # first_arts = all_articles_for_cust[firstidx][:,2]
            #
            # secondyr_from_date = datetime(2019, 8, 1) #Aug 01
            # secondyr_to_date = (secondyr_from_date + timedelta(days=120)).date()
            # secondidx = (times > secondyr_from_date.date()) & (times <= secondyr_to_date)
            # second_arts = all_articles_for_cust[secondidx][:,2]

            thirddyr_from_date = datetime(2020, 8, 1) #Aug 01
            thirddyr_to_date = (thirddyr_from_date + timedelta(days=90)).date()
            thirdidx = (times > thirddyr_from_date.date()) & (times <= thirddyr_to_date)
            third_arts = all_articles_for_cust[thirdidx][:,2]



            #customer_combined = np.insert(np.concatenate((first_arts, second_arts, third_arts), axis=0), 0, trs_val[pointer])
            if third_arts.shape[0] > 16:
                customer_combined = third_arts[-16:]
                customer_combined = np.insert(customer_combined, 0, trs_val[pointer])
                myfile.write(",".join(list(customer_combined.astype(str))) + "\n")

            # update vals
            cur_idx = end_idx
            pointer = pointer + 1
#buildCustTransactions()



class TextGenModel(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
      super().__init__(self)
      #self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
      self.gru = tf.keras.layers.GRU(rnn_units,
                                     return_sequences=True,
                                     return_state=True)
      self.dense_subseq = tf.keras.layers.Dense(input_shape=(10, 12, 100), units=256)
      self.dense = tf.keras.layers.Dense(vocab_size)

  def call(self, inputs, states=None, return_state=False, training=False):
      x = inputs
      #x = self.embedding(x, training=training)
      if states is None:
          states = self.gru.get_initial_state(x)
      x, states = self.gru(x, initial_state=states, training=training)
      x = self.dense(x, training=training)

      if return_state:
          return x, states
      else:
          return x



def load3DArray(filename, dim_b, dim_c):
    #return np.loadtxt("testing123.txt").reshape((dim_a, dim_b, dim_c))
    item = np.loadtxt(filename)
    return item.reshape(item.shape[0], dim_b, dim_c)


def make_dataset(data, targets):
    data = np.array(data, dtype=np.float32)
    tf.data.Dataset()
    ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=targets,
        sequence_length=1,
        sequence_stride=1,
        shuffle=True,
        batch_size=10,)

    return ds

def articlesToEmbeds(arr_of_article_ids):
    for item in range(len(arr_of_article_ids)):
        arr_of_article_ids[item] = np.where(articles == arr_of_article_ids[item])[0][0]
    return features[arr_of_article_ids.astype(int)][:,1:]

def articlesToIndices(arr_of_article_ids):
    for item in range(len(arr_of_article_ids)):
        arr_of_article_ids[item] = np.where(articles == arr_of_article_ids[item])[0][0]
    return arr_of_article_ids.astype(int)


def train_my_model(model, vectorized_train, vectorized_test, optimizer, loss_fn):
    loss_count = 0
    batch_size = 20
    pointer = 0


    for step in range(len(vectorized_train)// batch_size):
        train_batch = vectorized_train[pointer:(pointer+batch_size)].astype('float32')
        test_batch = vectorized_test[pointer:(pointer + batch_size)].astype('float32')

        with tf.GradientTape() as tape:

            logits = model(train_batch, training=True)  # Logits for this minibatch
            loss_value = loss_fn(test_batch, logits)
            loss_count = loss_count + loss_value

        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 100 batches.
        if step % 100 == 0:
            print("\nStep: " + str(step))
            print("Training loss (for  batch): " + str(loss_count / 100))
            loss_count = 0


    pointer = 0
    loss_count = 0

def loadData(vocab_size):
    all_train = None
    all_test = None


    #feedback_model = FeedBack(units=256, out_steps=OUT_STEPS, vocab_size=vocab_size)
    model = TextGenModel(vocab_size=vocab_size, embedding_dim=256, rnn_units=1024)


    #model = feedback_model

    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)
    # Instantiate a loss function.
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    pandas_file = pd.read_csv("nn_rich_train.csv", delimiter=",")
    data = pandas_file.to_numpy()

    train_data = data[:,1:9]
    test_data = data[:,9:]

    print("Prevectorization: " + str(train_data.shape))
    # vectorized_train = articlesToEmbeds(train_data.flatten().astype(str))
    # vectorized_test = articlesToIndices(test_data.flatten().astype(str))
    # np.savetxt("dump__vectorized_train.csv", vectorized_train, delimiter=',')
    # np.savetxt("dump__vectorized_test.csv", vectorized_test, delimiter=',')

    vectorized_train = np.loadtxt("dump__vectorized_train.csv",delimiter=',')
    vectorized_test = np.loadtxt("dump__vectorized_test.csv",delimiter=',')
    #model = tf.keras.models.load_model('checkpoint_epoch_15')

    vectorized_train = np.reshape(vectorized_train, (vectorized_train.shape[0]//8, 8, 11))
    vectorized_test = np.reshape(vectorized_test, (vectorized_test.shape[0] // 8, 8))

    for epoch in range(100):
        print("starting epoch " + str(epoch))


        if epoch == 30:
            optimizer.lr.assign(5e-3)
        elif epoch == 45:
            optimizer.lr.assign(1e-3)
        elif epoch == 55:
            optimizer.lr.assign(5e-4)
        elif epoch == 65:
            optimizer.lr.assign(1e-4)
        elif epoch == 75:
            optimizer.lr.assign(2e-5)

        print("running model....")
        train_my_model(model, vectorized_train, vectorized_test, optimizer, loss_fn)
        model.save("checkpoint_epoch_" + str(epoch))



articles_raw_data = pd.read_csv("./data/articles.csv")
articles_raw_data.head()
articles = np.array(articles_raw_data)[:,0].astype(str)
vocab_size = len(list(set(articles)))

embeddings_raw_data = pd.read_csv("./data/phrase_embeddings_pca_tiny.csv")
embeddings_raw_data.head()
features = np.array(embeddings_raw_data)

data = loadData(vocab_size)