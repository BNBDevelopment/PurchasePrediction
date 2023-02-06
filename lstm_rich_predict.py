import time
from datetime import datetime, timedelta

import tensorflow as tf
import pandas as pd
import numpy as np
import warnings

from keras.layers import TextVectorization

warnings.filterwarnings("ignore", message=r"Using", category=FutureWarning)



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
    with open("nn_rich_transactions.csv", "a") as myfile:
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
            if third_arts.shape[0] > 8:
                customer_combined = third_arts[-8:]
                customer_combined = np.insert(customer_combined, 0, trs_val[pointer])
                myfile.write(",".join(list(customer_combined.astype(str))) + "\n")

            # update vals
            cur_idx = end_idx
            pointer = pointer + 1

#buildCustTransactions()



articles_raw_data = pd.read_csv("./data/articles.csv")
articles_raw_data.head()
articles = np.array(articles_raw_data)[:,0].astype(int)

customers_raw_data = pd.read_csv("./data/customers.csv")
customers_raw_data.head()
customers = np.array(customers_raw_data)
customers = customers[:,0]

embeddings_raw_data = pd.read_csv("./data/phrase_embeddings_pca_tiny.csv")
embeddings_raw_data.head()
embeds = np.array(embeddings_raw_data)

model = tf.keras.models.load_model('checkpoint_epoch_30')



# transactions_raw_data = pd.read_csv("./data/transactions_train.csv")
# transactions_raw_data.head()
# transactions = np.array(transactions_raw_data)
# values, counts = np.unique(transactions[:,2], return_counts=True)
# top_ten_common = transactions[:,2][np.argpartition(-counts, kth=10)[:10]]
# del transactions

embeddings_raw_data = pd.read_csv("./data/phrase_embeddings_pca_tiny.csv")
embeddings_raw_data.head()
features = np.array(embeddings_raw_data)

def articlesToEmbeds(arr_of_article_ids):
    for item in range(len(arr_of_article_ids)):
        arr_of_article_ids[item] = np.where(articles == arr_of_article_ids[item])[0][0]
    return features[arr_of_article_ids.astype(int)][:,1:]

def articlesToIndices(arr_of_article_ids):
    for item in range(len(arr_of_article_ids)):
        arr_of_article_ids[item] = np.where(articles == arr_of_article_ids[item])[0][0]
    return arr_of_article_ids.astype(int)



# top_ten_common = np.array([478646001, 524825010, 661147004, 111593001, 306307076, 111593001, 546579001, 623434014, 399136009, 660519001])
with open("nn_rich_predictions_3_FINAL.csv", "w") as writefile:
    c_t = pd.read_csv("nn_rich_transactions.csv", delimiter=",").to_numpy()

    print("Starting predictions...")
    count = 0
    len_cust = len(customers)
    final_out = None
    customer_list = []
    batch_count = 1
    batch_size = 20
    batch = None

    print("Start Time = " + str(datetime.now().strftime("%H:%M:%S")))
    for line in c_t:
        count = count + 1
        customer_list.append(line[0])

        if batch_count % (batch_size) != 0:
            batch_count = batch_count + 1
            cust_t = articlesToEmbeds(line[1:].astype(int))
            if batch is None:
                batch = cust_t[None,...]
            else:
                batch = np.concatenate((batch, cust_t[None,...]), axis=0)
        else:
            cust_t = articlesToEmbeds(line[1:].astype(int))
            batch = np.concatenate((batch, cust_t[None, ...]), axis=0)
            batch_count = 1


            first_predicted_logits = model(batch.astype(np.float32))
            first_predicts = articles[np.argmax(first_predicted_logits, axis=-1)]
            second_vectorized = articlesToEmbeds(first_predicts.flatten().copy().astype(int))
            #second_vectorized = vectorize_layer(first_predicts.astype(str).flatten())
            second_vectorized = np.reshape(second_vectorized, (20,8,11))
            second_predicited_logits = model(second_vectorized.astype(np.float32))
            second_predicts = articles[np.argmax(second_predicited_logits, axis=-1)]

            temp_predicts = np.concatenate((first_predicts, second_predicts), axis=-1)
            batch = None


            for cust_index, cust_line in enumerate(temp_predicts):
                final_predicts = cust_line[np.sort(np.unique(cust_line, return_index=True, axis=-1)[1])]
                if final_predicts.shape[0] >= 12:
                    writefile.write(customer_list[cust_index] + "," + " ".join(list(final_predicts[:12].astype(str))) + "\n")

            customer_list = []