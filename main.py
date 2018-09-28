from dsg.loader import DataLoader

class Util():
    TRAIN_SESSION = "./data/train_session.csv"
    RANDOM_SUBMISSION = "./data/random_submission.csv"
    TEST_TRACKING = "./data/test_tracking.csv"
    PRODUCT_CATEGORY = "./data/productid_category.csv"

def main():
    print("LOADING TRAIN")
    df_train = DataLoader.load_data(Util.TRAIN_SESSION)
    print(df_train.head())
    print(df_train.describe())

    print("LOADING SUBMISSION")
    df_submission = DataLoader.load_data(Util.RANDOM_SUBMISSION)
    print(df_submission.head())
    print(df_submission.describe())

    print("LOADING TEST")
    df_test = DataLoader.load_data(Util.TEST_TRACKING)
    print(df_test.head())
    print(df_test.describe())

    print("LOADING PRODUCT")
    df_product = DataLoader.load_data(Util.PRODUCT_CATEGORY)
    print(df_product.head())
    print(df_product.describe())

    return

main()