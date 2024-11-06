#!/usr/bin/env python
# coding: utf-8

# In[3]:


import itertools
import pandas as pd
import time

def load_transactions_from_csv(file_path):
    df = pd.read_csv(file_path, header=None, skiprows=1)
    transactions = df[1].apply(lambda x: set(x.split(' , '))).tolist()
    return transactions
start_time = time.time()
def find_frequent_itemsets_bruteforce(transactions, min_support, max_k=3):
    unique_items = set(item for transaction in transactions for item in transaction)
    frequent_itemsets = []
    for k in range(1, max_k + 1):
        candidate_itemsets = generate_itemsets(unique_items, k)
        for itemset in candidate_itemsets:
            support = sum(1 for transaction in transactions if set(itemset).issubset(transaction))
            if support >= min_support:
                frequent_itemsets.append((set(itemset), support))
    return frequent_itemsets

def generate_itemsets(items, length):
    return list(itertools.combinations(items, length))

def generate_association_rules(frequent_itemsets, transactions, min_confidence):
    rules = []
    for itemset, support in frequent_itemsets:
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                antecedent = set(antecedent)
                consequent = itemset - antecedent
                antecedent_support = sum(1 for transaction in transactions if antecedent.issubset(transaction))
                if antecedent_support > 0:
                    confidence = support / antecedent_support
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))
    return rules

def process_dataset(file_path, min_support, min_confidence):
    transactions = load_transactions_from_csv(file_path)
    frequent_itemsets = find_frequent_itemsets_bruteforce(transactions, min_support)
    association_rules = generate_association_rules(frequent_itemsets, transactions, min_confidence)
    return frequent_itemsets, association_rules


file_paths = [
    r'C:\Users\Rushmitha K\Downloads\amazon_data.csv',
r'C:\Users\Rushmitha K\Downloads\costco_data.csv',
r'C:\Users\Rushmitha K\Downloads\best_buy_data.csv',
r'C:\Users\Rushmitha K\Downloads\walgreens_data.csv',
r'C:\Users\Rushmitha K\Downloads\walmart_data.csv'
]


min_support = 2
min_confidence = 0.5


for file_path in file_paths:
    frequent_itemsets, association_rules = process_dataset(file_path, min_support, min_confidence)
    print(f"Results for {file_path}:")
    print("Frequent Itemsets:", frequent_itemsets)
    print("Association Rules:", association_rules)
    print("\n")

brute_time = start_time - time.time()
print("Time taken: " )
print(brute_time * -1)


# In[11]:


import pandas as pd
from time import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def load_transactions(file_path):
    df = pd.read_csv(file_path)
    transactions = df['Filtered Transaction'].apply(lambda x: x.split(', ')).tolist()
    return transactions


def analyze_dataset(file_path, min_support, min_confidence):
    start_time = time() 

    transactions = load_transactions(file_path)
    
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    print(f"Input Transactions for {file_path.split('/')[-1]}:")  
    for t in transactions:
        print(t)
        
    print("\nGenerated Association Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence']])
    

    elapsed_time = time() - start_time
    print(f"\n\nTime taken for {file_path.split('/')[-1]}: {elapsed_time:.2f} seconds\n")


def main():
    
    file_paths = [
    'C:\Users\Rushmitha K\Downloads\amazon_data.csv',
    'C:\Users\Rushmitha K\Downloads\costco_data.csv',
    'C:\Users\Rushmitha K\Downloads\best_buy_data.csv',
    'C:\Users\Rushmitha K\Downloads\walgreens_data.csv',
    'C:\Users\Rushmitha K\Downloads\walmart_data.csv'
    ]
    

    
    min_support = float(input("Enter minimum support value (e.g., 0.05 for 5%): "))
    min_confidence = float(input("Enter minimum confidence value (e.g., 0.5 for 50%): "))

    for file_path in file_paths:
        analyze_dataset(file_path, min_support, min_confidence)

if __name__ == "__main__":
    main()


# In[12]:


import pandas as pd
from time import time
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules


def load_transactions(file_path):
    df = pd.read_csv(file_path)
    transactions = df['Filtered Transaction'].apply(lambda x: x.split(', ')).tolist()
    return transactions


def analyze_dataset(file_path, min_support, min_confidence):
    start_time = time()  

    transactions = load_transactions(file_path)
    
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
    
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    print(f"Input Transactions for {file_path.split('/')[-1]}:")  
    for t in transactions:
        print(t)
        
    print("\nGenerated Association Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence']])
    
    
    elapsed_time = time() - start_time
    print(f"\n\nTime taken for {file_path.split('/')[-1]}: {elapsed_time:.2f} seconds\n")


def main():
    
    datasets = {
        1:'C:\Users\Rushmitha K\Downloads\amazon_data.csv',
        2:'C:\Users\Rushmitha K\Downloads\costco_data.csv',
        3:'C:\Users\Rushmitha K\Downloads\best_buy_data.csv',
        4:'C:\Users\Rushmitha K\Downloads\walgreens_data.csv',
        5:'C:\Users\Rushmitha K\Downloads\walmart_data.csv''
    }
    
    
    print("Select the dataset(s) to analyze:")
    for key, value in datasets.items():
        print(f"{key} - {value.split('/')[-1]}")
    
    
    selections = input("Enter your choice(s) separated by commas (e.g., 1,3,5): ")
    selected_datasets = selections.split(',')
    
    
    min_support = float(input("Enter minimum support value (e.g., 0.05 for 5%): "))
    min_confidence = float(input("Enter minimum confidence value (e.g., 0.5 for 50%): "))

    
    for selection in selected_datasets:
        selected_path = datasets[int(selection)]
        analyze_dataset(selected_path, min_support, min_confidence)

if __name__ == "__main__":
    main()


# In[ ]:


from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd


def load_transactions(file_path):
    df = pd.read_csv(file_path)
    transactions = df['Filtered Transaction'].apply(lambda x: x.split(', ')).tolist()
    return transactions


def analyze_dataset_with_fpgrowth(file_path, min_support, min_confidence):
    transactions = load_transactions(file_path)
    
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    
    
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    

    print(f"Frequent Itemsets for {file_path.split('/')[-1]}:")
    print(frequent_itemsets)
    print("\nAssociation Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence']])
    print("\n\n")


def main():
    
    file_paths = [
   'C:\Users\Rushmitha K\Downloads\amazon_data.csv',
    'C:\Users\Rushmitha K\Downloads\costco_data.csv',
    'C:\Users\Rushmitha K\Downloads\best_buy_data.csv',
    'C:\Users\Rushmitha K\Downloads\walgreens_data.csv',
    'C:\Users\Rushmitha K\Downloads\walmart_data.csv'
    ]
    
    
    min_support = float(input("Enter minimum support value (e.g., 0.05 for 5%): "))
    min_confidence = float(input("Enter minimum confidence value (e.g., 0.5 for 50%): "))
    
    
    for file_path in file_paths:
        analyze_dataset_with_fpgrowth(file_path, min_support, min_confidence)

if __name__ == "__main__":
    main()


# In[8]:


import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
import time

def fetch_transactions(csv_path):
    """Loads transactions from a specified CSV file."""
    data = pd.read_csv(csv_path)['Filtered Transaction'].str.split(', ').tolist()
    return data
start_time3 = time.time()
def fp_growth_analysis(csv_path, support_level, confidence_level):
    """Performs FP-Growth analysis on the dataset for given support and confidence levels."""
    transactions = fetch_transactions(csv_path)
    

    encoder = TransactionEncoder()
    encoded_data = encoder.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(encoded_data, columns=encoder.columns_)
    
    
    itemsets = fpgrowth(df_encoded, min_support=support_level, use_colnames=True)
    
    
    rules = association_rules(itemsets, metric="confidence", min_threshold=confidence_level)
    
    
    print(f"\nFrequent Itemsets from {csv_path.split('/')[-1]}:")
    print(itemsets)
    print("\nDerived Association Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence']])
    print("\n")

def run_analysis():
    
    dataset_names = ['Amazon', 'Costco', 'Best Buy', 'Walgreens', 'Walmart']
    dataset_paths = { 
       1:'C:\Users\Rushmitha K\Downloads\amazon_data.csv',
        2:'C:\Users\Rushmitha K\Downloads\costco_data.csv',
        3:'C:\Users\Rushmitha K\Downloads\best_buy_data.csv',
        4:'C:\Users\Rushmitha K\Downloads\walgreens_data.csv',
        5:'C:\Users\Rushmitha K\Downloads\walmart_data.csv''

    }

    
    print("Select the dataset(s) to analyze:")
    for i, name in enumerate(dataset_names, 1):
        print(f"{i} - {name}")
    selection = input("Enter the numbers (separated by space) of the datasets to analyze (e.g., 1 3 for Amazon and Best Buy): ")

    selected_indices = selection.split()
    
    
    min_support = float(input("\nEnter minimum support value (e.g., 0.05 for 5%): "))
    min_confidence = float(input("Enter minimum confidence value (e.g., 0.5 for 50%): "))

    
    for index in selected_indices:
        if index.isdigit() and (1 <= int(index) <= len(dataset_paths)):
            fp_growth_analysis(dataset_paths[int(index)-1], min_support, min_confidence)
        else:
            print(f"Invalid selection: {index}")

if __name__ == "__main__":
    run_analysis()


# In[ ]:




