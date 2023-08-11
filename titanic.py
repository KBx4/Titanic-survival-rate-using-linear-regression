import pandas as pd
import tensorflow as tf


dftrain = pd.read_csv(r'C:\Users\Kabir\Downloads\train.csv')
dfeval = pd.read_csv(r'C:\Users\Kabir\Downloads\eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']
feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def make_input(data_df, label_df, shuffle = True, num_epochs = 10, batch_size = 32):
    def input():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input

train_input = make_input(dftrain, y_train)
eval_input = make_input(dfeval, y_eval, shuffle=False, num_epochs= 1)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
linear_est.train(train_input)
result = linear_est.evaluate(eval_input)
print(result['accuracy'])
result = list(linear_est.predict(eval_input))
print(dfeval.loc[1])
print(result[1]['probabilities'])
