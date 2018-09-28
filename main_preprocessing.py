from dsg.loader import DataLoader
from dsg.preprocessor import Preprocessor

SEQ_LENGTH = 10

def main():

    # load data
    df_train = DataLoader.load_train_data()

    # define preprocessor and transform data
    preprocessor = Preprocessor(seq_length=SEQ_LENGTH)
    X_train, y_train, X_test, y_test, X_val, y_val = preprocessor.transform(df_train)

    return

main()