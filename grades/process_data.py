#%%
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import glob

for h in glob.glob("*.html"):
    with open(h) as f:
        soup = BeautifulSoup(f.read())
    table = soup.find("table", {"id": "BodyTable"})
    table_body = table.find('tbody')
    rows = table_body.find_all('tr')
    headers = [e.text.strip() for e in table_body.find_all('th')]
    data = []
    for row in rows:
        cols = row.find_all('td')
        cols = [ele.text.strip() for ele in cols]
        if cols:
            data.append(cols[1:])
    df = pd.DataFrame(data, columns=headers[1:])
    df["exam_max"] = df[["final1", "final2", "final3"]].max(axis=1)
    # ex12 not in all years
    df = df[['ex1', 'ex2', 'ex3', 'ex4', 'ex5', 'ex6', 'ex7', 'ex8', 'ex9', 'ex10', 'ex11', 'exam_max']]
    df.to_csv(h.split(".")[0] + ".csv")
# %%
