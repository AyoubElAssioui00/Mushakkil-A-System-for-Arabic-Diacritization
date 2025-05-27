from Data.HMM import load_data, clean_text, load_pickle, split_train, extract_observation
import matplotlib.pyplot as plt



if __name__=="__main__":
    
    train_path=input("enter the input of training dataset")
    val_path=input("enter the input of validation dataset")
    test_path=input("enter the input of test dataset")
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    val_data=load_data(val_path)
    arabic_letters = load_pickle('constants/ARABIC_LETTERS_LIST.pickle')
    classes_list = load_pickle('constants/CLASSES_LIST.pickle')
    diacritics_list = load_pickle('constants/DIACRITICS_LIST.pickle')



    labels = ['Train', 'Validation', 'Test']
    sizes = [len(train_data), len(val_data), len(test_data)]
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, sizes, color=['skyblue', 'orange', 'green'])
    plt.title('Dataset Sizes')
    plt.ylabel('Number of Samples []')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 200, f'{yval}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('/home/assioui/Mushakkil/plots/dataset_sizes.png')



    # Count letter frequencies
    all_letters_in_train_data = {}
    for line in train_data:
        for char in line:
            if char.strip() == "":
                continue  # Skip whitespace
            if char not in all_letters_in_train_data:
                all_letters_in_train_data[char] = 0
            all_letters_in_train_data[char] += 1

    # Sort letters by frequency
    sorted_letters = sorted(all_letters_in_train_data.items(), key=lambda x: x[1], reverse=True)
    letters, frequencies = zip(*sorted_letters)

    # Plotting
    plt.figure(figsize=(14, 6))
    bars = plt.bar(letters, frequencies, color='skyblue')
    plt.title('Arabic Letter Frequency in Training Data')
    plt.xlabel('Letters')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('/home/assioui/Mushakkil/plots/Arabic_Letter_Frequency_in_Training_Data1.png')




    def filter(text):
        return ''.join([ch for ch in text if ch in arabic_letters])

    # Apply to train/val/test data
    filtered_train_data = [filter(line) for line in train_data]
    filtered_val_data = [filter(line) for line in val_data]
    filtered_test_data = [filter(line) for line in test_data]

    # Count letter frequencies
    all_letters_in_filtered_train_data = {}
    for line in filtered_train_data:
        for char in line:
            if char.strip() == "":
                continue  # Skip whitespace
            if char not in all_letters_in_filtered_train_data:
                all_letters_in_filtered_train_data[char] = 0
            all_letters_in_filtered_train_data[char] += 1

    # Sort letters by frequency
    sorted_letters = sorted(all_letters_in_filtered_train_data.items(), key=lambda x: x[1], reverse=True)
    letters, frequencies = zip(*sorted_letters)

    # Plotting
    plt.figure(figsize=(14, 6))
    bars = plt.bar(letters, frequencies, color='skyblue')
    plt.title('Arabic Letter Frequency in Training Data')
    plt.xlabel('Letters')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('/home/assioui/Mushakkil/plots/Arabic_Letter_Frequency_in_Training_Data2.png')

    


    chars = arabic_letters + ''.join(diacritics_list)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    def filter_arabic_range(text):
        return ''.join([ch for ch in text if ch in char_to_idx])

    # Apply to train/val/test data
    filtered_train_data = [filter_arabic_range(line) for line in train_data]
    filtered_val_data = [filter_arabic_range(line) for line in val_data]
    filtered_test_data = [filter_arabic_range(line) for line in test_data]

    # Count letter frequencies
    all_letters_in_filtered_train_data = {}
    for line in filtered_train_data:
        for char in line:
            if char.strip() == "":
                continue  # Skip whitespace
            if char not in all_letters_in_filtered_train_data:
                all_letters_in_filtered_train_data[char] = 0
            all_letters_in_filtered_train_data[char] += 1

    # Sort letters by frequency
    sorted_letters = sorted(all_letters_in_filtered_train_data.items(), key=lambda x: x[1], reverse=True)
    letters, frequencies = zip(*sorted_letters)

    # Plotting
    plt.figure(figsize=(14, 6))
    bars = plt.bar(letters, frequencies, color='skyblue')
    plt.title('Arabic Letter Frequency in Training Data')
    plt.xlabel('Letters')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('/home/assioui/Mushakkil/plots/Arabic_Letter_Frequency_in_Training_Data3.png')



    def count_each_dic(line):
        each = dict()
        for idx, char in enumerate(line):
            if char in diacritics_list:
                continue
            char_diac = ''
            if idx + 1 < len(line) and line[idx + 1] in diacritics_list:
                char_diac = line[idx + 1]
                if idx + 2 < len(line) and line[idx + 2] in diacritics_list and char_diac + line[idx + 2] in classes_list:
                    char_diac += line[idx + 2]
                elif idx + 2 < len(line) and line[idx + 2] in diacritics_list and line[idx + 2] + char_diac in classes_list:
                    char_diac = line[idx + 2] + char_diac
            try:
              each[char_diac] += 1
            except:
              each[char_diac] = 1

        return each

    diac_name = {'' : 'No Diacritic       ',
                  'َ' : 'Fatha              ',
                  'ً' : 'Fathatah           ',
                  'ُ' : 'Damma              ',
                  'ٌ' : 'Dammatan           ',
                  'ِ' : 'Kasra              ',
                  'ٍ' : 'Kasratan           ',
                  'ْ' : 'Sukun              ',
                  'ّ' : 'Shaddah            ',
                  'َّ' : 'Shaddah + Fatha    ',
                  'ًّ' : 'Shaddah + Fathatah ',
                  'ُّ' : 'Shaddah + Damma    ',
                  'ٌّ' : 'Shaddah + Dammatan ',
                  'ِّ' : 'Shaddah + Kasra    ',
                  'ٍّ' : 'Shaddah + Kasratan' }

    def filter_arabic_diacritics(data):
        counts = {key:0 for (_, key) in diac_name.items()}
        for line in data:
            each = count_each_dic(line)
            for key, value in each.items():
                counts[diac_name[key]] += value
        return counts
    

    counts = filter_arabic_diacritics(train_data)
    del counts['No Diacritic       ']
    # Sort diac by frequency
    sorted_diac = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    diac, frequencies = zip(*sorted_diac)

    # Plotting
    plt.figure(figsize=(14, 6))
    bars = plt.bar(diac, frequencies, color='skyblue')
    plt.title('Arabic diacritics Frequency in Training Data')
    plt.xlabel('diacriticss')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('/home/assioui/Mushakkil/plots/Arabic_Diacritics_Frequency_in_Training_Data.png')