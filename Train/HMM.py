from Algorithms.HMM import DiacriticHMM
from os import getcwd
from Data.HMM import load_data, clean_text, load_pickle, split_train, extract_observation


def train_test(train_path='train.txt',test_path='test.txt'):

    arabic_letters = load_pickle('constants/ARABIC_LETTERS_LIST.pickle')
    diacritics_list = load_pickle('constants/DIACRITICS_LIST.pickle')
    
    #load
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    
    #clean
    train_data=clean_text(train_data,arabic_letters,diacritics_list)
    test_data=clean_text(test_data,arabic_letters,diacritics_list)
    
    #Prepare for training
    train_data=split_train(train_data,arabic_letters)

    #Train the model
    hmm = DiacriticHMM()
    hmm.train(train_data)

    #save the model
    model_path = f"{getcwd()}/models/arabic_diacritization_hmm.pkl"
    hmm.save_model(model_path)
    
    #Prepare for testing
    test_observations=extract_observation(test_data,arabic_letters)
    test_data=split_train(test_data,arabic_letters)
    hmm_from_file = DiacriticHMM.load_from_file(model_path)
    predictions = hmm_from_file.predict(test_observations)

    #performance calculus
    der = hmm_from_file.calculate_diacritic_error_rate(predictions, test_data)
    print(f"\nDiacritic Error Rate: {der:.4f} ({der*100:.2f}%)")

if __name__=="__main__":
    
    train_path=input("Enter the path of your train dataset")
    
    test_path=input("Enter the path of your train dataset")

    train_test(train_path,test_path)