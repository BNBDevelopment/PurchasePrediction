from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from keras.layers import TextVectorization
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import pickle

embeddings_raw_data = pd.read_csv("./data/phrase_embeddings_pca_tiny.csv")
embeddings_raw_data.head()
features = np.array(embeddings_raw_data)


def createFiles():

    articles_raw_data = pd.read_csv("./data/articles.csv")
    articles_raw_data.head()
    articles = np.array(articles_raw_data)[:, 0]

    data = pd.read_csv("customer_with_all_transactions.csv", delimiter=",").to_numpy()
    customers = data[:, 0]
    labels = data[:, 1:]

    gnb = GaussianNB()
    # gnb = sklearn.naive_bayes.MultinomialNB()

    cust_list = []
    cust_lengths = []
    combined = None
    all_lbls = None
    count = 0

    with open("labels_asdasdcust.csv", "w") as cust_file:
        with open("labels_asdasd.csv", "w") as labels_file:
            with open("train_inputs_asdasdasd.csv", "w") as train_inputs_file:
                for label_idx in range(labels.shape[0]):
                    label = labels[label_idx]
                    customer = customers[label_idx]
                    count = count + 1
                    if count % 100 == 0:
                        print("Count: " + str(count) + "     out of: " + str(len(labels)))
                        labels_file.write(",".join(all_lbls.flatten().astype(str).tolist()) + "\n")
                        train_inputs_file.write(",".join(combined.flatten().astype(str).tolist()) + "\n")
                        cust_file.write(",".join(cust_list) + "," + ",".join(cust_list))
                        combined = None
                        all_lbls = None
                        cust_list = []
                        cust_lengths = []
                    new_labels = label[label != 0]

                    for item in range(len(new_labels)):
                        new_labels[item] = np.where(articles == new_labels[item])[0][0]

                    indexes = new_labels.astype(int)
                    vals = features[indexes]
                    if len(vals.shape) == 0:
                        vals = np.zeros((1,12))
                    elif len(vals.shape) == 1:
                        vals = vals[None, ...]
                    cust_rep = np.sum(vals[:,1:], axis=0)

                    if combined is None:
                        combined = cust_rep[None, ...].repeat(len(new_labels), axis=0)
                    else:
                        combined = np.concatenate((combined, cust_rep[None, ...].repeat(len(new_labels), axis=0)), axis=0)

                    cust_list.append(customer)
                    cust_lengths.append(len(new_labels))

                    if all_lbls is None:
                        all_lbls = new_labels[..., None]
                    else:
                        all_lbls = np.concatenate((all_lbls, new_labels[..., None]), axis=0)


articles_raw_data = pd.read_csv("./data/articles.csv")
articles_raw_data.head()
articles = np.array(articles_raw_data)[:, 0].astype(int)
def articlesToIndices(arr_of_article_ids):
    for item in range(len(arr_of_article_ids)):
        arr_of_article_ids[item] = np.where(articles == arr_of_article_ids[item])[0][0]
    return arr_of_article_ids

#createFiles()


top_ten_common = np.array([478646001, 524825010, 661147004, 111593001, 306307076, 111593001, 546579001, 623434014, 399136009, 660519001])
gnb = GaussianNB()



def fitFast():
    customers_raw_data = pd.read_csv("./data/customers.csv")
    customers_raw_data.head()
    customers = np.array(customers_raw_data)
    customers = customers[:, 0]
    del customers_raw_data

    articles_raw_data = pd.read_csv("./data/articles.csv")
    articles_raw_data.head()
    articles = np.array(articles_raw_data)[:, 0].astype(int)
    del articles_raw_data




    x_vals = np.loadtxt("_rf_x_vals")
    x_lbl = np.loadtxt("_rf_x_lbl")
    test_vals = np.loadtxt("_rf_test_vals")
    test_lbl = np.loadtxt("_rf_test_lbl")



    print("RF Classifier Fitting...")
    gnb.fit(x_vals, x_lbl.ravel())


    with open('gnb_v3.pkl', 'wb') as rfc_file:
        pickle.dump(gnb, rfc_file)


    y_pred = gnb.predict(test_vals)
    print("RFC Accuracy:", metrics.accuracy_score(test_lbl, y_pred))



def predict():
    customers_raw_data = pd.read_csv("./data/customers.csv")
    customers_raw_data.head()
    customers = np.array(customers_raw_data)
    customers = customers[:, 0]

    with open("predictions_nb_small.txt", "w") as writefile:
        c_t = pd.read_csv("customer_with_all_transactions.csv", delimiter=",").to_numpy()

        print("Starting predictions...")
        count = 0
        len_cust = len(customers)
        final_out = None
        customer_list = []
        batch_count = 1
        batch_size = 100
        batch = None
        no_hist_list = []
        print("Start Time = " + str(datetime.now().strftime("%H:%M:%S")))
        for customer in customers:
            count = count + 1
            customer_list.append(customer)

            cust_t = c_t[np.where(c_t[:, 0] == customer)]
            if cust_t.shape[0] > 0:
                cust_t = cust_t[0]

            new_labels = cust_t[cust_t != 0][1:]

            for item in range(len(new_labels)):
                new_labels[item] = np.where(articles == new_labels[item])[0][0]

            indexes = new_labels.astype(int)
            vals = features[indexes]
            if len(vals.shape) == 0:
                no_hist_list.append(customer)
                vals = np.zeros((1, 12))
            elif len(vals.shape) == 1:
                vals = vals[None, ...]
            cust_rep = np.sum(vals[:, 1:], axis=0)


            if batch_count % (batch_size) != 0:
                batch_count = batch_count + 1
                if batch is None:
                    batch = cust_rep[None, ...]
                else:
                    batch = np.concatenate((batch, cust_rep[None, ...]), axis=0)
            else:
                batch = np.concatenate((batch, cust_rep[None, ...]), axis=0)
                batch_count = 1

                final_predicts = gnb.predict_proba(batch)

                for idx in range(batch_size):
                    if customer_list[idx] in no_hist_list:
                        write_string = customer_list[idx] + ",0" + " 0".join(top_ten_common.astype(str).tolist()) + "\n"
                    else:
                        customer_predictions = final_predicts[idx]
                        twleve_max_indices = np.argpartition(customer_predictions, -12, axis=-1)[-12:]
                        #twelve_max_probs = customer_predictions[twleve_max_indices]
                        final_articles = articles[gnb.classes_[twleve_max_indices]]
                        write_string = customer_list[idx] + ",0" + " 0".join(final_articles.astype(str).tolist()) + "\n"
                    writefile.write(write_string)

                no_hist_list = []
                customer_list = []

                if count % 1000 == 0:
                    print("Processed " + str(count) + "/" + str(len_cust))




