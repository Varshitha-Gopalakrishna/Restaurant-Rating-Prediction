def clean_data(df):
    df = df.copy()

    # Drop unwanted columns
    df.drop(['url', 'dish_liked', 'phone', 'menu_item'], axis=1, inplace=True)

    # Rename columns
    df.rename({'approx_cost(for two people)': 'approx_cost_for_2_people',
               'listed_in(type)': 'listed_in_type',
               'listed_in(city)': 'listed_in_city'}, axis=1, inplace=True)

    # Clean 'votes' and 'approx_cost_for_2_people'
    df['votes'] = df['votes'].astype(int)
    df['approx_cost_for_2_people'] = df['approx_cost_for_2_people'].apply(lambda x: int(x.replace(',', '')) if isinstance(x, str) else x)

    # Clean and convert 'rate'
    df = df[df['rate'] != 'NEW']
    df = df[df['rate'] != '-']
    df['rate'] = df['rate'].apply(lambda x: x.replace('/5', '').strip() if isinstance(x, str) else x).astype(float)

    return df
