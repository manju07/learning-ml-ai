# Pandas — Data Analysis Library

## Table of Contents
- [Introduction](#introduction)
- [Series](#series)
- [DataFrame](#dataframe)
- [Reading and Writing Data](#reading-and-writing-data)
- [Selecting and Filtering](#selecting-and-filtering)
- [Data Cleaning](#data-cleaning)
- [Data Transformation](#data-transformation)
- [Aggregation and GroupBy](#aggregation-and-groupby)
- [Merging and Joining](#merging-and-joining)
- [Time Series](#time-series)
- [Performance Tips](#performance-tips)

---

## Introduction

Pandas provides two primary data structures:
- **Series** — 1D labeled array
- **DataFrame** — 2D labeled table (think spreadsheet or SQL table)

```python
import pandas as pd
import numpy as np
print(pd.__version__)
```

---

## Series

A Series is a one-dimensional labeled array.

```python
import pandas as pd

# Create from list
s = pd.Series([10, 20, 30, 40], name="numbers")
print(s)
# 0    10
# 1    20
# 2    30
# 3    40
# Name: numbers, dtype: int64

# Custom index
s = pd.Series([10, 20, 30], index=["a", "b", "c"])
print(s["b"])    # 20

# From dict
s = pd.Series({"apple": 1.5, "banana": 0.75, "cherry": 2.0})
print(s.index)    # Index(['apple', 'banana', 'cherry'])
print(s.values)   # [1.5  0.75  2.  ]

# Attributes
print(s.dtype)    # float64
print(s.shape)    # (3,)
print(s.size)     # 3
print(len(s))     # 3

# Operations
print(s * 2)             # multiply all prices
print(s[s > 1.0])        # filter
print(s.sort_values())   # sort
print(s.sort_index())    # sort by index
print(s.describe())      # summary stats
```

---

## DataFrame

A DataFrame is a 2D labeled data structure.

### Creating DataFrames

```python
import pandas as pd
import numpy as np

# From dict of lists
df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "Dave"],
    "age":    [30, 25, 35, 28],
    "salary": [90000, 75000, 110000, 82000],
    "dept":   ["Engineering", "HR", "Engineering", "Marketing"],
})

# From list of dicts
df = pd.DataFrame([
    {"name": "Alice", "age": 30},
    {"name": "Bob",   "age": 25},
])

# From NumPy array
arr = np.random.randn(4, 3)
df  = pd.DataFrame(arr, columns=["A", "B", "C"])

# From CSV (see Reading section)
df = pd.read_csv("data.csv")
```

### Basic Inspection

```python
df = pd.read_csv("employees.csv")

df.head(5)         # first 5 rows
df.tail(3)         # last 3 rows
df.shape           # (rows, cols)
df.ndim            # 2
df.size            # total elements
df.columns         # column names
df.index           # row labels
df.dtypes          # data type of each column
df.info()          # concise summary with memory usage
df.describe()      # statistical summary (numeric cols)
df.describe(include="all")   # include categorical
df.value_counts()  # count unique combinations
```

### Accessing Columns and Rows

```python
# Column access
df["name"]          # Series — single column
df[["name", "age"]] # DataFrame — multiple columns

# Row access — always prefer loc/iloc
df.loc[0]           # by label (index value)
df.loc[0:2]         # inclusive slice by label
df.loc[[0, 2, 4]]   # by list of labels
df.iloc[0]          # by integer position
df.iloc[0:3]        # exclusive slice by position
df.iloc[[0, 2, 4]]  # by list of positions

# Cell access
df.loc[0, "name"]          # label-based
df.iloc[0, 0]              # position-based
df.at[0, "name"]           # fast scalar loc
df.iat[0, 0]               # fast scalar iloc

# Row + column slice
df.loc[0:2, "name":"age"]  # inclusive
df.iloc[0:3, 0:2]          # exclusive
```

---

## Reading and Writing Data

### CSV

```python
# Read
df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv",
    sep=",",               # delimiter
    header=0,              # row to use as header (None = no header)
    names=["a", "b", "c"], # column names (if no header)
    index_col="id",        # column to use as index
    usecols=["name", "age"],  # only load these columns
    dtype={"age": int, "salary": float},
    parse_dates=["date"],  # parse date columns
    na_values=["N/A", "missing", ""],
    encoding="utf-8",
    nrows=100,             # only read first 100 rows
    skiprows=[1, 2],       # skip specific rows
    chunksize=10000,       # read in chunks
)

# Write
df.to_csv("output.csv", index=False)   # index=False avoids extra column
df.to_csv("output.csv", sep="\t")      # tab-separated
```

### Excel

```python
# pip install openpyxl xlrd
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
df = pd.read_excel("data.xlsx", sheet_name=0)       # first sheet by index

# Write
df.to_excel("output.xlsx", sheet_name="Results", index=False)

# Write multiple sheets
with pd.ExcelWriter("output.xlsx", engine="openpyxl") as writer:
    df1.to_excel(writer, sheet_name="Sheet1", index=False)
    df2.to_excel(writer, sheet_name="Sheet2", index=False)
```

### JSON

```python
df = pd.read_json("data.json")
df = pd.read_json("data.json", orient="records")  # list of dicts

df.to_json("output.json", orient="records", indent=2)
```

### SQL

```python
import sqlite3

conn = sqlite3.connect("database.db")

df = pd.read_sql("SELECT * FROM users WHERE age > 25", conn)
df = pd.read_sql_table("users", conn)   # requires sqlalchemy

df.to_sql("users", conn, if_exists="replace", index=False)
# if_exists: 'fail', 'replace', 'append'

conn.close()
```

### Parquet (Columnar Format — Recommended for Large Data)

```python
# pip install pyarrow
df.to_parquet("data.parquet", index=False)
df = pd.read_parquet("data.parquet")
df = pd.read_parquet("data.parquet", columns=["name", "age"])
```

---

## Selecting and Filtering

```python
import pandas as pd

df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol", "Dave", "Eve"],
    "age":    [30, 25, 35, 28, 22],
    "salary": [90000, 75000, 110000, 82000, 65000],
    "dept":   ["Eng", "HR", "Eng", "Mktg", "HR"],
})

# Boolean filtering
adults = df[df["age"] >= 28]
high_earners = df[df["salary"] > 80000]

# Multiple conditions
eng_seniors = df[(df["dept"] == "Eng") & (df["age"] > 28)]
hr_or_mktg  = df[df["dept"].isin(["HR", "Mktg"])]
not_hr      = df[~(df["dept"] == "HR")]

# query() — more readable for complex conditions
result = df.query("dept == 'Eng' and salary > 80000")
result = df.query("age > @threshold", threshold=25)  # local variable

# String operations
df[df["name"].str.startswith("A")]
df[df["name"].str.contains("al", case=False)]
df[df["name"].str.len() > 3]

# isin / between
df[df["age"].between(25, 30)]          # inclusive
df[df["dept"].isin(["Eng", "HR"])]

# nlargest / nsmallest
df.nlargest(3, "salary")
df.nsmallest(2, "age")
```

---

## Data Cleaning

### Missing Values

```python
df = pd.DataFrame({
    "A": [1, None, 3, None, 5],
    "B": [None, 2, 3, 4, 5],
    "C": [1, 2, None, 4, 5],
})

# Detection
df.isna()                   # boolean DataFrame
df.isnull()                 # alias
df.notna()
df.isna().sum()             # count missing per column
df.isna().sum().sum()       # total missing
df.isna().mean() * 100      # % missing per column

# Dropping
df.dropna()                 # drop rows with ANY missing value
df.dropna(how="all")        # drop rows where ALL values are missing
df.dropna(subset=["A","B"]) # drop rows with missing in specific cols
df.dropna(axis=1)           # drop columns with missing values
df.dropna(thresh=3)         # keep rows with at least 3 non-null values

# Filling
df.fillna(0)                # fill all NaN with 0
df.fillna({"A": 0, "B": df["B"].mean()})  # per-column fill
df.fillna(method="ffill")   # forward fill (propagate previous value)
df.fillna(method="bfill")   # backward fill
df["A"].fillna(df["A"].mean(), inplace=True)

# Interpolation
df["A"].interpolate(method="linear")
df["A"].interpolate(method="polynomial", order=2)
```

### Duplicates

```python
df.duplicated()                       # boolean Series — True for duplicates
df.duplicated(subset=["name"])        # based on specific columns
df.duplicated(keep="first")           # keep first occurrence
df.drop_duplicates()                  # remove duplicate rows
df.drop_duplicates(subset=["name"])   # based on column
df.drop_duplicates(keep="last")       # keep last occurrence
```

### Data Type Conversion

```python
df["age"] = df["age"].astype(int)
df["salary"] = df["salary"].astype(float)
df["date"] = pd.to_datetime(df["date"])
df["category"] = df["category"].astype("category")  # memory efficient

# Numeric conversion
df["value"] = pd.to_numeric(df["value"], errors="coerce")  # NaN on failure

# String cleaning
df["name"] = df["name"].str.strip()
df["name"] = df["name"].str.lower()
df["name"] = df["name"].str.replace(r"\s+", " ", regex=True)

# Replace specific values
df.replace({"N/A": None, "unknown": None})
df["status"].replace({"Y": True, "N": False})
```

### Renaming

```python
df.rename(columns={"old_name": "new_name"})
df.rename(columns=str.lower)   # apply function to all names
df.columns = df.columns.str.lower().str.replace(" ", "_")
df.index = df.index + 1        # shift index
df.reset_index(drop=True)      # reset to 0-based integer index
df.set_index("id")             # use column as index
```

---

## Data Transformation

### Apply and Map

```python
import pandas as pd

df = pd.DataFrame({
    "name":   ["Alice", "Bob", "Carol"],
    "salary": [90000, 75000, 110000],
    "score":  [85, 72, 91],
})

# apply — along axis
df["salary_k"] = df["salary"].apply(lambda x: x / 1000)
df["grade"] = df["score"].apply(
    lambda s: "A" if s >= 90 else "B" if s >= 80 else "C"
)

# apply on multiple columns (row-wise)
df["combined"] = df.apply(
    lambda row: f"{row['name']}: {row['salary_k']:.0f}K",
    axis=1  # axis=1 = apply across columns (row by row)
)

# map — element-wise on Series (rename values)
df["dept_code"] = df["dept"].map({"Eng": "E", "HR": "H", "Mktg": "M"})

# applymap (deprecated 2.1+, use map instead)
# df.map(lambda x: x * 2)   # apply to every element in DataFrame

# transform — like apply but keeps original shape (for group operations)
df["salary_zscore"] = df.groupby("dept")["salary"].transform(
    lambda s: (s - s.mean()) / s.std()
)
```

### Adding and Removing Columns

```python
# Add column
df["bonus"] = df["salary"] * 0.1
df.assign(bonus=df["salary"] * 0.1, total=lambda df: df["salary"] + df["bonus"])

# Remove column
df.drop("bonus", axis=1, inplace=True)
df.drop(columns=["bonus", "total"])

# Insert at specific position
df.insert(2, "bonus", df["salary"] * 0.1)

# Computed columns with eval (fast for simple expressions)
df.eval("bonus = salary * 0.1", inplace=True)
df.eval("total = salary + bonus")
```

### Sorting

```python
df.sort_values("age")                          # ascending
df.sort_values("age", ascending=False)         # descending
df.sort_values(["dept", "salary"], ascending=[True, False])  # multi-key
df.sort_index()                                 # sort by index
df.sort_values("salary").reset_index(drop=True)
```

### String Methods (`.str`)

```python
s = pd.Series(["  Alice Smith  ", "BOB JONES", "carol@example.com"])

s.str.strip()
s.str.lower()
s.str.upper()
s.str.title()
s.str.replace("@", "_at_")
s.str.split(" ")               # returns list
s.str.split(" ", expand=True)  # returns DataFrame
s.str.contains("@")            # boolean mask
s.str.startswith("A")
s.str.len()
s.str.extract(r"(\w+)@(\w+)")  # regex capture groups → DataFrame
s.str.cat(sep=", ")            # concatenate all elements
```

---

## Aggregation and GroupBy

```python
import pandas as pd

df = pd.DataFrame({
    "dept":   ["Eng", "HR", "Eng", "Mktg", "HR", "Eng"],
    "name":   ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"],
    "salary": [90000, 75000, 110000, 82000, 65000, 95000],
    "score":  [85, 72, 91, 78, 88, 80],
    "year":   [2020, 2019, 2021, 2020, 2022, 2018],
})

# Basic aggregation
print(df["salary"].mean())
print(df["salary"].agg(["mean", "std", "min", "max"]))

# GroupBy
grouped = df.groupby("dept")

# Single aggregation
print(grouped["salary"].mean())
print(grouped["salary"].agg("sum"))

# Multiple aggregations
print(grouped["salary"].agg(["mean", "sum", "count"]))

# Different aggs per column
result = grouped.agg({
    "salary": ["mean", "sum", "max"],
    "score":  ["mean", "std"],
})
print(result)
print(result.columns)  # MultiIndex columns

# Named aggregation (cleaner)
result = grouped.agg(
    avg_salary=("salary", "mean"),
    max_salary=("salary", "max"),
    headcount=("name", "count"),
)
print(result)

# Custom aggregation function
result = grouped["salary"].agg(lambda s: s.max() - s.min())

# transform — return same-shape result aligned with original
df["dept_avg_salary"] = grouped["salary"].transform("mean")
df["salary_rank"]     = grouped["salary"].transform("rank", ascending=False)

# filter — select groups based on a condition
high_paying_depts = grouped.filter(lambda g: g["salary"].mean() > 80000)

# apply — most flexible (can return anything)
def top_earners(group):
    return group.nlargest(2, "salary")

result = grouped.apply(top_earners)
```

### Pivot Tables

```python
# Pivot table — like Excel pivot tables
pivot = df.pivot_table(
    values="salary",
    index="dept",
    columns="year",
    aggfunc="mean",
    fill_value=0,
    margins=True,    # add row/column totals
)

# Simple pivot (no aggregation — unique index/column pairs required)
pivot = df.pivot(index="name", columns="dept", values="salary")

# Melt — inverse of pivot (wide → long format)
df_wide = pd.DataFrame({
    "name": ["Alice", "Bob"],
    "jan":  [90, 80],
    "feb":  [85, 75],
    "mar":  [95, 90],
})
df_long = df_wide.melt(
    id_vars="name",
    value_vars=["jan", "feb", "mar"],
    var_name="month",
    value_name="score"
)
# name  month  score
# Alice jan    90
# ...

# Stack / Unstack
stacked   = df.stack()    # columns → rows (wide → long)
unstacked = df.unstack()  # rows → columns (long → wide)

# crosstab — frequency table
pd.crosstab(df["dept"], df["year"], margins=True)
```

---

## Merging and Joining

```python
import pandas as pd

employees = pd.DataFrame({
    "emp_id": [1, 2, 3, 4],
    "name":   ["Alice", "Bob", "Carol", "Dave"],
    "dept_id": [10, 20, 10, 30],
})

departments = pd.DataFrame({
    "dept_id": [10, 20, 40],
    "dept_name": ["Engineering", "HR", "Finance"],
})

# merge — like SQL JOIN
# Inner join (default)
inner = pd.merge(employees, departments, on="dept_id")
# Only rows where dept_id exists in BOTH

# Left join
left = pd.merge(employees, departments, on="dept_id", how="left")
# All employees, NaN for unmatched departments

# Right join
right = pd.merge(employees, departments, on="dept_id", how="right")

# Outer join
outer = pd.merge(employees, departments, on="dept_id", how="outer")

# Different column names
pd.merge(employees, departments,
         left_on="dept_id", right_on="dept_id",
         suffixes=("_emp", "_dept"))

# join() — index-based join
df1 = employees.set_index("emp_id")
df2 = departments.set_index("dept_id")

# concat — combine DataFrames
# Vertical (stack rows)
combined = pd.concat([df1, df2], ignore_index=True)

# Horizontal (side by side)
combined = pd.concat([df1, df2], axis=1)

# With keys for MultiIndex
combined = pd.concat([df1, df2], keys=["emp", "dept"])
```

---

## Time Series

```python
import pandas as pd
import numpy as np

# Create date range
dates = pd.date_range("2024-01-01", periods=365, freq="D")
# Frequencies: D=daily, W=weekly, M=month end, MS=month start, H=hourly, T=minutely

ts = pd.Series(np.random.randn(365), index=dates, name="value")

# Parse dates
df = pd.read_csv("data.csv", parse_dates=["date"])
df["date"] = pd.to_datetime(df["date"])
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")

# Date components
df["year"]    = df["date"].dt.year
df["month"]   = df["date"].dt.month
df["day"]     = df["date"].dt.day
df["weekday"] = df["date"].dt.day_name()
df["quarter"] = df["date"].dt.quarter

# Resampling (aggregating by time period)
weekly  = ts.resample("W").mean()
monthly = ts.resample("ME").sum()
yearly  = ts.resample("YE").std()

# Custom aggregation
monthly_stats = ts.resample("ME").agg({
    "value": ["mean", "std", "min", "max"]
})

# Rolling window
rolling_mean = ts.rolling(window=7).mean()    # 7-day rolling mean
rolling_std  = ts.rolling(window=30).std()    # 30-day rolling std
ewm_mean     = ts.ewm(span=7).mean()          # exponentially weighted

# Shifting
ts.shift(1)         # shift 1 period forward (creates 1 NaN at start)
ts.shift(-1)        # shift backward
ts.pct_change()     # percentage change from previous period
ts.diff()           # absolute difference from previous period

# Time zone handling
ts_utc = ts.tz_localize("UTC")
ts_ny  = ts_utc.tz_convert("America/New_York")
```

---

## Performance Tips

```python
import pandas as pd
import numpy as np

# 1. Use categorical dtype for repeated string columns
df["dept"] = df["dept"].astype("category")
# Memory: O(n * string_length) → O(n) + unique strings

# 2. Use appropriate numeric dtypes
df["age"]    = df["age"].astype("int8")      # if age < 128
df["salary"] = df["salary"].astype("float32")  # if precision not critical

# 3. Avoid iterrows() — use vectorized or apply
# Slow:
for idx, row in df.iterrows():
    df.at[idx, "bonus"] = row["salary"] * 0.1

# Fast:
df["bonus"] = df["salary"] * 0.1

# 4. Use query() and eval() for large DataFrames
df.query("age > 25 and dept == 'Eng'")  # faster than boolean indexing
df.eval("bonus = salary * 0.1")

# 5. Read large files in chunks
chunks = []
for chunk in pd.read_csv("huge.csv", chunksize=100_000):
    filtered = chunk[chunk["age"] > 25]
    chunks.append(filtered)
df = pd.concat(chunks, ignore_index=True)

# 6. Use parquet over CSV for large datasets
df.to_parquet("data.parquet")
df = pd.read_parquet("data.parquet", columns=["name", "age"])

# 7. Check memory usage
print(df.info(memory_usage="deep"))
print(df.memory_usage(deep=True).sum() / 1e6, "MB")

# 8. pandas-profiling for EDA
# pip install ydata-profiling
from ydata_profiling import ProfileReport
profile = ProfileReport(df)
profile.to_file("report.html")
```
