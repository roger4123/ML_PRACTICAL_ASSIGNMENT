import pandas as pd
import numpy as np

def prepare_data(file_path):
    print("1. Loading data...")
    try:
        df = pd.read_csv(file_path)
    except:
        raise FileNotFoundError(f"{file_path} not found.")
    print(f"    Initial data loaded: {df.shape[0]} lines.")

    print("2. Selecting relevant columns...")
    cols_to_keep = ['id_bon', 'data_bon', 'retail_product_name', 'SalePriceWithVAT']
    
    missing_cols = [c for c in cols_to_keep if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns from CSV file: {missing_cols}")
        
    df = df[cols_to_keep].copy()

    # Convert price to numerical value
    df['SalePriceWithVAT'] = df['SalePriceWithVAT'].astype(float)


    print("3. Temporal data processing...")
    df['data_bon'] = pd.to_datetime(df['data_bon'])
    
    df['hour'] = df['data_bon'].dt.hour
    df['day_of_week'] = df['data_bon'].dt.dayofweek # 0=Luni, 6=Duminica
    
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)


    print("4. Agregate per receipt")
    
    # General statistics per receipt
    # Luăm 'first' la timp (toate produsele de pe un bon au aceeași oră)
    # Facem sumă la preț și count la produse
    df_agg = df.groupby('id_bon').agg({
        'data_bon': 'first',
        'hour': 'first',
        'day_of_week': 'first',
        'is_weekend': 'first',
        'SalePriceWithVAT': 'sum', # total value
        'retail_product_name': 'count' # nr. of products
    }).rename(columns={
        'SalePriceWithVAT': 'total_value',
        'retail_product_name': 'cart_size'
    })


    print("5. Pivoting (transforming products into columns)...")
    
    # Index = id_bon, Columns = nume produse, Values = de cate ori apare produsul
    df_products = pd.crosstab(df['id_bon'], df['retail_product_name'])
    
    # Opțional: Dacă vrei doar 0/1 (fără cantități), decomentează linia de mai jos:
    # df_products = (df_products > 0).astype(int)


    print("6. Finalizing...")
    
    df_final = df_agg.join(df_products)
    df_final = df_final.sort_values('data_bon')

    print(f"    Final dataset: {df_final.shape[0]} receipts x {df_final.shape[1]} columns.")
    return df_final