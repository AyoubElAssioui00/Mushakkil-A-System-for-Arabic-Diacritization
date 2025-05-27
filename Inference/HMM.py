from Algorithms.HMM import DiacriticHMM
from os import getcwd
from Data.HMM import clean_text, load_pickle, extract_observation, split_train


arabic_letters = load_pickle('constants/ARABIC_LETTERS_LIST.pickle')
diacritics_list = load_pickle('constants/DIACRITICS_LIST.pickle')
def predict(test_data):

    test_data=clean_text(test_data,arabic_letters,diacritics_list)

    test_observations=extract_observation(test_data,arabic_letters)

    test_data=split_train(test_data,arabic_letters)

    hmm_from_file = DiacriticHMM.load_from_file(f"{getcwd()}/models/arabic_diacritization_hmm.pkl")

    predictions = hmm_from_file.predict(test_observations)

    for i, prediction in enumerate(predictions):
        print(f"The diacritiation of your text:  ")
        for obs, pred_diacritic in prediction:
            print(f"{obs}{pred_diacritic}", end="")
if __name__=="__main__":
    ch=[input("enter your sentence")]
    predict(ch)
     