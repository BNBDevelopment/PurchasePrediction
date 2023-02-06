from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from keras.layers import TextVectorization
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import pickle

def predictFast():
    customers_raw_data = pd.read_csv("./data/customers.csv")
    customers_raw_data.head()
    customers = np.array(customers_raw_data)
    customers = customers[:, 0]
    del customers_raw_data

    articles_raw_data = pd.read_csv("./data/articles.csv")
    articles_raw_data.head()
    articles = np.array(articles_raw_data)[:, 0].astype(int)
    del articles_raw_data

    transactions_raw_data = pd.read_csv("./data/transactions_train.csv")
    transactions_raw_data.head()
    transactions = np.array(transactions_raw_data)
    values, counts = np.unique(transactions[:, 2], return_counts=True)
    top_twelve_common = transactions[:, 2][np.argpartition(-counts, kth=12)[:12]]
    del transactions
    del transactions_raw_data

    gnb = None
    with open('medium_nb.pkl', 'rb') as fid:
        gnb = pickle.load(fid)

    #top_twelve_common = np.array([478646001, 524825010, 661147004, 111593001, 306307076, 111593001, 546579001, 623434014, 399136009, 660519001, 478646001, 478646001])

    customer_indices = pd.read_csv("final_customers.csv").to_numpy()
    customer_names_per_line = customer_indices[:, :99]
    customer_lengths = customer_indices[:, 99:]

    #customer_data = pd.read_csv("final_train_inputs.csv").to_numpy()

    with open("nb_02.csv", "w") as writefile:
        with open("final_train_inputs.csv", "r") as train_inputs_file:
            line_number = -1

            combined_vals = None
            combined_lengths = None
            combined_names = None

            for values_line in train_inputs_file:
                line_number = line_number + 1
                print("ON: " + str(line_number) + "   out of: " + str(customer_lengths.shape[0]))

                lengths_line = customer_lengths[line_number]
                names_line = customer_names_per_line[line_number]

                pointer = 0
                split_vals = values_line.split(",")
                comma_vals = np.split(np.array(split_vals), len(split_vals) // 30)
                #line_preds = gnb.predict_proba(np.array(comma_vals).astype(np.float64))

                vals_line = np.array(comma_vals).astype(np.float64)

                if combined_vals is None: combined_vals = vals_line
                else: combined_vals = np.concatenate((combined_vals, vals_line), axis=0)

                if combined_lengths is None: combined_lengths = lengths_line
                else: combined_lengths = np.concatenate((combined_lengths, lengths_line), axis=0)

                if combined_names is None: combined_names = names_line
                else: combined_names = np.concatenate((combined_names, names_line), axis=0)

                if line_number % 8 == 0:
                    line_preds = gnb.predict_proba(combined_vals)

                    counter = 0
                    for point_val in combined_lengths:
                        end_point = pointer + point_val
                        if point_val == 0:
                            if len(str(combined_names[counter])) > 10:
                                write_string = combined_names[counter] + ",0" + " 0".join(top_twelve_common.astype(str).tolist()) + "\n"
                        else:
                            block_for_customer = line_preds[pointer:end_point, :]
                            if len(str(combined_names[counter])) < 10:
                                write_string = str(combined_names[counter]) + ",0" + " 0".join(
                                    top_twelve_common.astype(str).tolist()) + "\n"
                                pointer = end_point
                            elif len(block_for_customer.shape) < 2:
                                write_string = combined_names[counter] + ",0" + " 0".join(
                                    top_twelve_common.astype(str).tolist()) + "\n"
                                pointer = end_point
                            else:
                                #block_for_customer = np.array(comma_vals[pointer:end_point], :).astype(np.float64)
                                #customer_predictions = np.sum(gnb.predict_proba(block_for_customer), axis=0)
                                customer_predictions = np.sum(block_for_customer, axis=0)
                                twleve_max_indices = np.argpartition(customer_predictions, -12, axis=-1)[-12:]
                                twelve_max_probs = customer_predictions[twleve_max_indices]
                                final_articles = articles[gnb.classes_[twleve_max_indices]]
                                write_string = combined_names[counter] + ",0" + " 0".join(final_articles.astype(str).tolist()) + "\n"
                                pointer = end_point
                        counter = counter + 1
                        writefile.write(write_string)
                    combined_vals = None
                    combined_lengths = None
                    combined_names = None
            #line_preds = gnb.predict_proba(combined_vals)

            counter = 0
            for point_val in combined_lengths:
                end_point = pointer + point_val
                if point_val == 0:
                    if len(str(combined_names[counter])) > 10:
                        write_string = combined_names[counter] + ",0" + " 0".join(
                            top_twelve_common.astype(str).tolist()) + "\n"
                else:
                    block_for_customer = line_preds[pointer:end_point, :]
                    if len(str(combined_names[counter])) < 10:
                        write_string = str(combined_names[counter]) + ",0" + " 0".join(
                            top_twelve_common.astype(str).tolist()) + "\n"
                        pointer = end_point
                    elif len(block_for_customer.shape) < 2:
                        write_string = combined_names[counter] + ",0" + " 0".join(
                            top_twelve_common.astype(str).tolist()) + "\n"
                        pointer = end_point
                    else:
                        # block_for_customer = np.array(comma_vals[pointer:end_point], :).astype(np.float64)
                        # customer_predictions = np.sum(gnb.predict_proba(block_for_customer), axis=0)
                        customer_predictions = np.sum(block_for_customer, axis=0)
                        twleve_max_indices = np.argpartition(customer_predictions, -12, axis=-1)[-12:]
                        twelve_max_probs = customer_predictions[twleve_max_indices]
                        final_articles = articles[gnb.classes_[twleve_max_indices]]
                        write_string = combined_names[counter] + ",0" + " 0".join(
                            final_articles.astype(str).tolist()) + "\n"
                        pointer = end_point
                counter = counter + 1
                writefile.write(write_string)
            combined_vals = None
            combined_lengths = None
            combined_names = None


predictFast()