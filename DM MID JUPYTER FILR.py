#!/usr/bin/env python
# coding: utf-8

# In[32]:


pip install mlxtend pandas


# In[33]:


import itertools
import time
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules


# In[34]:


# Function to calculate support for an itemset.

def calculate_support(transactions, itemset):
    return sum(1 for transaction in transactions if set(itemset).issubset(set(transaction))) / len(transactions)


# In[75]:


# Brute Force algorithm to generate frequent itemset.
def brute_force_frequent_itemsets(transactions, min_support):
    items = sorted(set(item for transaction in transactions for item in transaction)) # get the unique items from the transactions.
    frequent_itemsets = []
    itemset_size = 1
    
    while True:
        candidate_itemsets = list(itertools.combinations(items, itemset_size))     # Generate all combinations of itemset from current size.  
        current_frequent_itemsets = []
        
        for itemset in candidate_itemsets:
            support = calculate_support(transactions, itemset)
            if support >= min_support:
                current_frequent_itemsets.append((itemset, support))
        
        if not current_frequent_itemsets:
            break
        
        frequent_itemsets.extend(current_frequent_itemsets)
        itemset_size += 1

    return frequent_itemsets


# In[76]:


def generate_association_rules(frequent_itemsets, transactions, min_confidence):
    rules = []
    for itemset, itemset_support in frequent_itemsets:
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for antecedent in itertools.combinations(itemset, i):
                    consequent = tuple(item for item in itemset if item not in antecedent)
                    antecedent_support = calculate_support(transactions, antecedent)
                    confidence = itemset_support / antecedent_support
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence, itemset_support))
    return rules


# Prepare Transaction DataFrame Function

# In[77]:


def prepare_transaction_df(transactions):
    items = sorted(set(item for transaction in transactions for item in transaction))
    return pd.DataFrame([[item in transaction for item in items] for transaction in transactions], columns=items).astype(bool)


# Run All Algorithms Function

# In[78]:


def run_all_algorithms(transactions, min_support, min_confidence):
    transaction_df = prepare_transaction_df(transactions)

    # Brute force algorithm
    print("\nRunning Brute Force Algorithm...")
    start_time = time.time()
    frequent_itemsets_brute = brute_force_frequent_itemsets(transactions, min_support)
    rules_brute = generate_association_rules(frequent_itemsets_brute, transactions, min_confidence)
    brute_time = time.time() - start_time
    print(f"Brute Force Time: {brute_time:.4f} seconds")
    print_results("Brute Force", frequent_itemsets_brute, rules_brute)

    # Apriori Algorithm
    print("\nRunning Apriori Algorithm...")
    start_time = time.time()
    try:
        frequent_itemsets_apriori = apriori(transaction_df, min_support=min_support, use_colnames=True)
        apriori_time = time.time() - start_time
        
        if frequent_itemsets_apriori.empty:
            print("Apriori did not find any frequent itemsets.")
        else:
            rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=min_confidence)
            print(f"Apriori Time: {apriori_time:.4f} seconds")
            print_results("Apriori", frequent_itemsets_apriori, rules_apriori)
    except Exception as e:
        print(f"An error occurred during Apriori algorithm execution: {e}")
        apriori_time = time.time() - start_time

    # FP-Growth Algorithm
    print("\nRunning FP-Growth Algorithm...")
    start_time = time.time()
    try:
        frequent_itemsets_fpgrowth = fpgrowth(transaction_df, min_support=min_support, use_colnames=True)
        fpgrowth_time = time.time() - start_time
        
        if frequent_itemsets_fpgrowth.empty:
            print("FP-Growth did not find any frequent itemsets.")
        else:
            rules_fpgrowth = association_rules(frequent_itemsets_fpgrowth, metric="confidence", min_threshold=min_confidence)
            print(f"FP-Growth Time: {fpgrowth_time:.4f} seconds")
            print_results("FP-Growth", frequent_itemsets_fpgrowth, rules_fpgrowth)
    except Exception as e:
        print(f"An error occurred during FP-Growth algorithm execution: {e}")
        fpgrowth_time = time.time() - start_time
    
    return brute_time, apriori_time, fpgrowth_time


# In[79]:


def read_csv_and_prepare_transactions(transaction_file, itemset_file):
    try:
        df_trans = pd.read_csv(transaction_file)
        df_items = pd.read_csv(itemset_file)
        
        if 'Item #' in df_items.columns and 'Item Name' in df_items.columns:
            item_map = dict(zip(df_items['Item #'], df_items['Item Name']))
        else:
            item_map = None
        
        transactions = []
        for _, row in df_trans.iterrows():
            transaction = []
            for item in row:
                if isinstance(item, str):
                    items = [i.strip() for i in item.split(',') if i.strip()]
                    transaction.extend(items)
                elif pd.notna(item):
                    transaction.append(str(item))
            if transaction:
                transactions.append(transaction)
        
        return transactions
    
    except Exception as e:
        print(f"Error reading the CSV files: {str(e)}")
        raise


# In[80]:


def print_results(algorithm_name, frequent_itemsets, rules):
    print(f"\n{algorithm_name} Results:")
    print("Frequent Itemsets:")
    if isinstance(frequent_itemsets, pd.DataFrame):
        for _, row in frequent_itemsets.iterrows():
            print(f"Items: {set(row['itemsets'])}, Support: {row['support']*100:.2f}%")
    else:
        for itemset, support in frequent_itemsets:
            print(f"Items: {set(itemset)}, Support: {support*100:.2f}%")
    
    print("\nAssociation Rules:")
    if isinstance(rules, pd.DataFrame):
        for _, rule in rules.iterrows():
            print(f"Rule: {set(rule['antecedents'])} -> {set(rule['consequents'])}")
            print(f"Confidence: {rule['confidence']*100:.2f}%, Support: {rule['support']*100:.2f}%")
            print()
    else:
        for antecedent, consequent, confidence, support in rules:
            print(f"Rule: {set(antecedent)} -> {set(consequent)}")
            print(f"Confidence: {confidence*100:.2f}%, Support: {support*100:.2f}%")
            print()


# In[83]:


def main():
    # Define available stores and their corresponding files
    stores = {
        "1": ("amazon", "amazon_data.csv", "amazon_data.csv"),
        "2": ("bestbuy", "best_buy_data.csv", "best_buy_data.csv"),
        "3": ("Costco", "Costco_Transaction.csv.csv", "Costco_Transaction.csv.csv"),
        "4": ("walgreens", "walgreens_Transaction.csv.csv", "walgreens_Transaction.csv.csv"),
        "5": ("walmart", "walmart_data.csv", "walmart_data.csv")
    }
    
    print("\nAvailable stores:")
    for key, (store_name, _, _) in stores.items():
        print(f"{key}. {store_name}")

    # Get user input for store selection.
    while True:
        choice = input("Enter the number of the store you want to analyze: ")
        if choice in stores:
            store_name, transaction_file, itemset_file = stores[choice]
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

    # Read and prepare transaction data.
    try:
        transactions = read_csv_and_prepare_transactions(transaction_file, itemset_file)
        if not transactions:
            print("No valid transactions found. Please check your CSV files.")
            return
    except Exception as e:
        print(f"Error reading the CSV files: {e}")
        print("Please ensure that the CSV files are properly formatted.")
        return

# Get user input for minimum support and minimum confidence.
    while True:
        try:
            min_support = float(input("Enter the minimum support (as a percentage between 0 and 100): "))
            if 0 <= min_support <= 100:
                min_support /= 100  # Convert to decimal
                break
            else:
                print("Please enter a value between 0 and 100.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            min_confidence = float(input("Enter the minimum confidence (as a percentage between 0 and 100): "))
            if 0 <= min_confidence <= 100:
                min_confidence /= 100  # Convert to decimal
                break
            else:
                print("Please enter a value between 0 and 100.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    brute_time, apriori_time, fpgrowth_time = run_all_algorithms(transactions, min_support, min_confidence)

    print(f"\nExecution Times:")
    print(f"Brute Force: {brute_time:.4f} seconds")
    print(f"Apriori: {apriori_time:.4f} seconds")
    print(f"FP-Growth: {fpgrowth_time:.4f} seconds")

    # Print the fastest algorithm.
    fastest_algorithm = min((brute_time, 'Brute Force'), (apriori_time, 'Apriori'), (fpgrowth_time, 'FP-Growth'))[1]
    print(f"\nThe fastest algorithm is: {fastest_algorithm}")

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




