import argparse
import datasets
import pandas
import transformers
import tensorflow as tf
import numpy
from tensorflow.keras import backend as K
from sklearn.metrics import multilabel_confusion_matrix
from tensorflow import keras

# use the tokenizer from DistilRoBERTa
tokenizer = transformers.AutoTokenizer.from_pretrained("distilroberta-base")


def tokenize(examples):
    """Converts the text of each example to "input_ids", a sequence of integers
    representing 1-hot vectors for each token in the text"""
    return tokenizer(examples["text"], truncation=True, max_length=64,
                     padding="max_length",return_tensors="tf")


def to_bow(example):
    """Converts the sequence of 1-hot vectors into a single many-hot vector"""
    vector = numpy.zeros(shape=(tokenizer.vocab_size,))
    vector[example["input_ids"]] = 1
    return {"input_bow": vector}

def custom_loss(y_true,y_pred):
    wp={0:3.2264,
             1:5.1963,
             2:4.3980,
             3:6.2480,
             4:104.9666,
             5:87.4722,
             6:23.1544}
    wn={
        0:1,
        1:0.5532,
        2:0.5641,
        3:0.5434,
        4:0.5023,
        5:0.5,
        6:0.51
    }
    loss = float(0)   
    for i, key in enumerate(wp.keys()):
        first_term = wp[key] * y_true[i] * K.log(y_pred[i] + K.epsilon())
        second_term = wn[key] * (1 - y_true[i]) * K.log(1 - y_pred[i] + K.epsilon())
        loss -= (first_term + second_term)
    return loss
    
    print(standard_loss(y_true,y_pred))
    weighted_loss = standard_loss(y_true, y_pred) * penalty_weights
    return weighted_loss
keras.utils.get_custom_objects()['custom_loss'] = custom_loss
def train(model_path="model", train_path="train.csv", dev_path="dev.csv"):
    # load the CSVs into Huggingface datasets to allow use of the tokenizer
    hf_dataset = datasets.load_dataset("csv", data_files={
        "train": train_path, "validation": dev_path})

    # the labels are the names of all columns except the first
    labels = hf_dataset["train"].column_names[1:]

    def gather_labels(example):
        """Converts the label columns into a list of 0s and 1s"""
        # the float here is because F1Score requires floats
        print([float(example[l]) for l in labels])
        return {"labels": [float(example[l]) for l in labels]}

    # convert text and labels to format expected by model
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    hf_dataset = hf_dataset.map(gather_labels)
    #hf_dataset = hf_dataset.map(to_bow)

    # convert Huggingface datasets to Tensorflow datasets
    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns="input_ids",
        label_cols="labels",
        batch_size=16,
        shuffle=True)
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns="input_ids",
        label_cols="labels",
        batch_size=16)
    early_stop=tf.keras.callbacks.EarlyStopping(
        monitor="val_f1_score",
        patience=2,
        mode="max"
    )
    # define a model with a single fully connected layer
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(tokenizer.vocab_size,64))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32),
                backward_layer=tf.keras.layers.GRU(32,go_backwards=True)))
    model.add(tf.keras.layers.Dense(16, activation="selu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(
            7,
            activation='sigmoid'
            ))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)
    # specify compilation hyperparameters
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001,clipnorm=1.0),
        loss= tf.keras.losses.binary_crossentropy,
        metrics=[tf.keras.metrics.F1Score(average="weighted", threshold=0.5)])
    # fit the model to the training data, monitoring F1 on the dev data
    model.fit(
        train_dataset,
        epochs=100,
        batch_size=32,
        validation_data=dev_dataset,
        callbacks=[tensorboard_callback,
            tf.keras.callbacks.ModelCheckpoint(
                filepath=model_path,
                monitor="val_f1_score",
                mode="max",
                save_best_only=True),early_stop],
            class_weight={0:3.2264,
             1:5.1963,
             2:4.3980,
             3:6.2480,
             4:104.9666,
             5:87.4722,
             6:23.1544})

def predict(model_path="model", input_path="dev.csv"):

    # load the saved model
    model = tf.keras.models.load_model(model_path)

    # load the data for prediction
    # use Pandas here to make assigning labels easier later
    df = pandas.read_csv(input_path)

    # create input features in the same way as in train()
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize, batched=True)
    #hf_dataset = hf_dataset.map(to_bow)
    tf_dataset = hf_dataset.to_tf_dataset(
        columns="input_ids",
        batch_size=16)

    # generate predictions from model
    predictions = numpy.where(model.predict(tf_dataset) > 0.5, 1, 0)
    c=multilabel_confusion_matrix(predictions,df[['admiration', 'amusement', 'gratitude', 'love', 'pride', 'relief', 'remorse']])
    for i, cm in enumerate(c):
        print(f"Confusion matrix for label {i + 1}:\n{cm}")
    # assign predictions to label columns in Pandas data frame"""
    df.iloc[:, 1:] = predictions

    # write the Pandas dataframe to a zipped CSV file
    df.to_csv("submission.zip", index=False, compression=dict(
        method='zip', archive_name=f'submission.csv'))


if __name__ == "__main__":
    # parse the command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()

    # call either train() or predict()
    globals()[args.command]()
