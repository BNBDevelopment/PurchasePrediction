from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization
from tensorflow.python.client import device_lib

def buildCustTransactions():
    transactions_raw_data = pd.read_csv("./data/transactions_train.csv")
    transactions_raw_data.head()
    transactions = np.array(transactions_raw_data)

    sorted_transactions = transactions[transactions[:, 1].argsort()]
    del transactions
    trs_val, trs_index = np.unique(sorted_transactions[:, 1], return_counts=True)

    cur_idx = 0
    pointer = 0

    all_combined = None

    with open("cust_and_trans.csv", "a") as myfile:
        while pointer < len(trs_index):
            if pointer % 100 == 0:
                print("Loaded: " + str(pointer) + " / " + str(len(trs_index)))

            end_idx = cur_idx + trs_index[pointer]
            all_articles_for_cust = sorted_transactions[cur_idx:end_idx, :]

            times = all_articles_for_cust[:, 0].astype(np.datetime64)
            thirddyr_from_date = datetime(2020, 8, 1)  # Aug 01
            thirddyr_to_date = (thirddyr_from_date + timedelta(days=90)).date()
            thirdidx = (times > thirddyr_from_date.date()) & (times <= thirddyr_to_date)
            third_arts = all_articles_for_cust[thirdidx][:, 2]

            customer_combined = third_arts
            customer_combined = np.insert(customer_combined, 0, trs_val[pointer])
            myfile.write(",".join(list(customer_combined.astype(str))) + "\n")

            # update vals
            cur_idx = end_idx
            pointer = pointer + 1
#buildCustTransactions()

articles_raw_data = pd.read_csv("./data/articles.csv")
articles_raw_data.head()
articles = np.array(articles_raw_data)

def articlesToIndices(arr_of_article_ids):
    for item in range(len(arr_of_article_ids)):
        arr_of_article_ids[item] = np.where(articles == arr_of_article_ids[item])[0][0]
    return arr_of_article_ids.astype(int)

def transactionsToBuckets():
    customers_raw_data = pd.read_csv("./data/customers.csv")
    customers_raw_data.head()
    customers = np.array(customers_raw_data)

    customers_ids = customers[:, 0]
    customers_age = customers[:, -2]
    del customers_raw_data

    customers_age[np.where(customers_age.astype(str) == 'nan')[0]] = '32'
    ages = customers_age.astype(int)

    group_1_articles = None
    group_2_articles = None
    group_3_articles = None
    group_4_articles = None
    group_5_articles = None
    group_6_articles = None
    group_7_articles = None
    group_8_articles = None

    tran_cust_ids = np.loadtxt("dump_____tran_cust_ids.csv", delimiter=",", dtype=str)
    tran_articles = np.loadtxt("dump_____tran_articles.csv", delimiter=",", dtype=str)

    id_age = np.concatenate((customers_ids[..., None], customers_age[..., None]), axis=-1)
    id_article = np.concatenate((tran_cust_ids[..., None], tran_articles[..., None]), axis=-1)

    # build two dataframe from set1 and set2
    df1 = pd.DataFrame(columns=['x0', 'x1'])
    df1['x0'] = [x[0] for x in id_age]
    df1['x1'] = [x[1] for x in id_age]

    df2 = pd.DataFrame(columns=['x0', 'x2'])
    df2['x0'] = [x[0] for x in id_article]
    df2['x2'] = [x[1] for x in id_article]

    numpy_combined = pd.merge(df1, df2, on=['x0'], how='left').to_numpy()
    trans_with_age = numpy_combined[np.where(numpy_combined.astype(str)[:, 2] != 'nan')]

    for temp_age in range(15, 120):
        cust_wth_age = trans_with_age[np.where(trans_with_age[:,1].astype(int) == temp_age)]
        if cust_wth_age.shape[0] > 0:
            this_articles = cust_wth_age[:,2]
            if temp_age > 0 and temp_age <= 19:
                if group_1_articles is None:
                    group_1_articles = this_articles
                else:
                    group_1_articles = np.concatenate((group_1_articles, this_articles))
            elif temp_age > 19 and temp_age <= 24:
                if group_2_articles is None:
                    group_2_articles = this_articles
                else:
                    group_2_articles = np.concatenate((group_2_articles, this_articles))
            elif temp_age > 24 and temp_age <= 30:
                if group_3_articles is None:
                    group_3_articles = this_articles
                else:
                    group_3_articles = np.concatenate((group_3_articles, this_articles))
            elif temp_age > 30 and temp_age <= 35:
                if group_4_articles is None:
                    group_4_articles = this_articles
                else:
                    group_4_articles = np.concatenate((group_4_articles, this_articles))
            elif temp_age > 35 and temp_age <= 45:
                if group_5_articles is None:
                    group_5_articles = this_articles
                else:
                    group_5_articles = np.concatenate((group_5_articles, this_articles))
            elif temp_age > 45 and temp_age <= 55:
                if group_6_articles is None:
                    group_6_articles = this_articles
                else:
                    group_6_articles = np.concatenate((group_6_articles, this_articles))
            elif temp_age > 55 and temp_age <= 65:
                if group_7_articles is None:
                    group_7_articles = this_articles
                else:
                    group_7_articles = np.concatenate((group_7_articles, this_articles))
            elif temp_age > 65:
                if group_8_articles is None:
                    group_8_articles = this_articles
                else:
                    group_8_articles = np.concatenate((group_8_articles, this_articles))



    print("Eval 1...")
    values, counts = np.unique(group_1_articles, return_counts=True)
    group_1_top_twelve = group_1_articles[np.argpartition(-counts, kth=12)[:12]]
    np.savetxt("group_1.csv", group_1_top_twelve, delimiter=",", fmt='%s')

    print("Eval 2...")
    values, counts = np.unique(group_2_articles, return_counts=True)
    group_2_top_twelve = group_2_articles[np.argpartition(-counts, kth=12)[:12]]
    np.savetxt("group_2.csv", group_2_top_twelve, delimiter=",", fmt='%s')

    print("Eval 3...")
    values, counts = np.unique(group_3_articles, return_counts=True)
    group_3_top_twelve = group_3_articles[np.argpartition(-counts, kth=12)[:12]]
    np.savetxt("group_3.csv", group_3_top_twelve, delimiter=",", fmt='%s')

    values, counts = np.unique(group_4_articles, return_counts=True)
    group_4_top_twelve = group_4_articles[np.argpartition(-counts, kth=12)[:12]]
    np.savetxt("group_4.csv", group_4_top_twelve, delimiter=",", fmt='%s')

    values, counts = np.unique(group_5_articles, return_counts=True)
    group_5_top_twelve = group_5_articles[np.argpartition(-counts, kth=12)[:12]]
    np.savetxt("group_5.csv", group_5_top_twelve, delimiter=",", fmt='%s')

    values, counts = np.unique(group_6_articles, return_counts=True)
    group_6_top_twelve = group_6_articles[np.argpartition(-counts, kth=12)[:12]]
    np.savetxt("group_6.csv", group_6_top_twelve, delimiter=",", fmt='%s')

    print("Eval 7...")
    values, counts = np.unique(group_7_articles, return_counts=True)
    group_7_top_twelve = group_7_articles[np.argpartition(-counts, kth=12)[:12]]
    np.savetxt("group_7.csv", group_7_top_twelve, delimiter=",", fmt='%s')

    values, counts = np.unique(group_8_articles, return_counts=True)
    group_8_top_twelve = group_8_articles[np.argpartition(-counts, kth=12)[:12]]
    np.savetxt("group_8.csv", group_8_top_twelve, delimiter=",", fmt='%s')


def combine_predicts():
    nb_predictions = pd.read_csv("some_predictions_local_FINAL.csv").to_numpy().astype(str)
    lstm_predictions = pd.read_csv("nn_rich_predictions_5.csv").to_numpy().astype(str)
    customers_raw_data = pd.read_csv("./data/customers.csv")
    customers_raw_data.head()
    customers = np.array(customers_raw_data)

    customers_ids = customers[:, 0]
    customers_age = customers[:, -2]
    customers_age[np.where(customers_age.astype(str) == 'nan')[0]] = '32'
    customers_age = customers_age.astype(int)
    del customers_raw_data
    del customers

    group_1_articles = np.loadtxt("group_1.csv", delimiter=",", dtype=str)
    group_2_articles = np.loadtxt("group_2.csv", delimiter=",", dtype=str)
    group_3_articles = np.loadtxt("group_3.csv", delimiter=",", dtype=str)
    group_4_articles = np.loadtxt("group_4.csv", delimiter=",", dtype=str)
    group_5_articles = np.loadtxt("group_5.csv", delimiter=",", dtype=str)
    group_6_articles = np.loadtxt("group_6.csv", delimiter=",", dtype=str)
    group_7_articles = np.loadtxt("group_7.csv", delimiter=",", dtype=str)
    group_8_articles = np.loadtxt("group_8.csv", delimiter=",", dtype=str)

    age_lists = [" 0".join(group_1_articles), " 0".join(group_2_articles), " 0".join(group_3_articles), " 0".join(group_4_articles),
                          " 0".join(group_5_articles), " 0".join(group_6_articles), " 0".join(group_7_articles), " 0".join(group_8_articles)]

    age_map = {}
    for i in range(1, 120):
        if i > 0 and i <= 19:
            age_map[i] = age_lists[0]
        elif i > 19 and i <= 24:
            age_map[i] = age_lists[1]
        elif i > 24 and i <= 30:
            age_map[i] = age_lists[2]
        elif i > 30 and i <= 35:
            age_map[i] = age_lists[3]
        elif i > 35 and i <= 45:
            age_map[i] = age_lists[4]
        elif i > 45 and i <= 55:
            age_map[i] = age_lists[5]
        elif i > 55 and i <= 65:
            age_map[i] = age_lists[6]
        else:
            age_map[i] = age_lists[7]


    # build two dataframe from set1 and set2
    df1 = pd.DataFrame(columns=['c_id', 'age'])
    df1['c_id'] = [x.strip("[").strip("]").strip("'") for x in customers_ids.astype(str)]
    df1['age'] = [age_map[x] for x in customers_age.astype(int)]
    df1['final'] = ['nan' for x in customers_age]

    nb_predictions = np.unique(nb_predictions, axis=0)

    df2 = pd.DataFrame(columns=['c_id', 'lstm'])
    df2['c_id'] = [x[0] for x in lstm_predictions.astype(str)]
    df2['lstm'] = [x[1] for x in lstm_predictions.astype(str)]

    df3 = pd.DataFrame(columns=['c_id', 'nb'])
    df3['c_id'] = [x[0] for x in nb_predictions.astype(str)]
    df3['nb'] = [x[1] for x in nb_predictions.astype(str)]

    one_two = pd.merge(df1, df2, on=['c_id'], how='left')
    one_two_three = pd.merge(one_two, df3, on=['c_id'], how='left')
    #one_two_three['final'] = one_two_three['final'].where(one_two_three['lstm'] != 'Nan', one_two_three['lstm'])

    with open("_ALL_FINAL.csv", "w") as writefile:
        for line in one_two_three.to_numpy().astype(str):
            if line[3] != 'nan':
                writefile.write(line[0] + "," + line[3] + "\n")
            elif line[4] != 'nan' and line[4] != "0478646001 0524825010 0661147004 0111593001 0306307076 0111593001 0546579001 0623434014 0399136009 0633130002 0660519001 0681358001":
                writefile.write(line[0] + "," + line[4] + "\n")
            else:
                writefile.write(line[0] + ",0" + str(line[1]) + "\n")

combine_predicts()