'''
This script is to get the schema and columns of the parquet files in 
the data directory. 
'''
import pyarrow.parquet as pq

print("-----------------COMPETITIONS PARQUET--------------------")
# Inspect the file WITHOUT loading it into memory
file = pq.read_metadata("../data/comps.parquet")

print(file.num_rows)        # how many rows
print(file.num_row_groups)  # how it's chunked internally
print(file.serialized_size) # compressed size in bytes
print("-------------------------------------------------")

# See the schema (column names + types) — free, reads no data
schema = pq.read_schema("../data/comps.parquet")
print(schema)

print("-----------------EVENTS PARQUET--------------------")
file2 = pq.read_metadata("../data/events.parquet")

print(file2.num_rows)        # how many rows
print(file2.num_row_groups)  # how it's chunked internally
print(file2.serialized_size) # compressed size in bytes
print("-------------------------------------------------")

# See the schema (column names + types) — free, reads no data
schema2 = pq.read_schema("../data/events.parquet")
print(schema2)

print("-----------------MATCHES PARQUET--------------------")
file3 = pq.read_metadata("../data/matches.parquet")

print(file3.num_rows)        # how many rows
print(file3.num_row_groups)  # how it's chunked internally
print(file3.serialized_size) # compressed size in bytes
print("-------------------------------------------------")

# See the schema (column names + types) — free, reads no data
schema3 = pq.read_schema("../data/matches.parquet")
print(schema2)