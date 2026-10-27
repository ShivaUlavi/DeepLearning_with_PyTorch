import io
import os
import unicodedata
import string
import glob
import torch
import random


# alphabet small + capital letters + " .,;'"
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

#Turn Unicode string to ASCII
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD',s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

def load_data():
    #build category_lines dictionary and a list of names per language
    category_lines = {}
    all_categories = []

    def find_files(path):
        return glob.glob(path)

    def read_lines(filename):
        #read a file and split into lines
        lines = io.open(filename, encoding = 'utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]

    for filename in find_files('./data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    return category_lines, all_categories


##converting letters into one-hot vectors of size (line_length, 1, n_letters), we add 1 because pytorch assumes everything is in 
#batches & we'll be using a batch size of 1
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)

#Example
def letter_to_tensor(letter):
    tensor = torch.zeros(1, N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor


def line_to_tensor(line):
    tensor = torch.zeros(len(line),1, N_LETTERS)
    for i, letter in enumerate(line):
        tensor[i][0][letter_to_index(letter)] = 1
    return tensor

def random_training_example(category_lines, all_categories):

    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]

    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype = torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor


if __name__ == '__main__':
    print(ALL_LETTERS)
    print(unicode_to_ascii('Ślusàrski'))

    category_lines, all_categories = load_data()
    print(category_lines['Russian'][-5:])

    print(letter_to_tensor('Z'))
    print(line_to_tensor('Allah').size())
    