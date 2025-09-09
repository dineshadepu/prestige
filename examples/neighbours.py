import numpy as np


def int_coord(coord):
    return int(np.floor(coord / spacing))


def hash_coords(x, y, z):
    xi = int_coord(x)
    yi = int_coord(y)
    zi = int_coord(z)
    h = (xi * 92837111) ^ (yi * 689287499) ^ (zi * 283923481)
    return abs(h) % table_size


spacing = 0.1
x = np.arange(0, 1.01, spacing)  # 1.01 to include 1.0 due to floating point precision
y = np.arange(0, 1.01, spacing)
z = np.arange(0, spacing + 0.001, spacing)

# x = np.arange(0, 4. * spacing, spacing)  # 1.01 to include 1.0 due to floating point precision
# y = np.arange(0, 4. * spacing, spacing)
# z = np.arange(0, spacing + 0.001, spacing)

# Create meshgrid
x, y, z = np.meshgrid(x, y, z)
x = x.ravel()
y = y.ravel()
z = z.ravel()
# print(len(x))
# print(x, y)

# create the table
num_objects = len(x)
table_size = 10 * num_objects
cell_start = np.zeros(table_size + 1, dtype=np.int64)
cell_entries = np.zeros(num_objects, dtype=np.int64)
query_ids = np.zeros(num_objects, dtype=np.int64)
query_size = 0


# Step 1: Clear previous data
cell_start.fill(0)
cell_entries.fill(0)

# Step 2: Count number of objects per cell
for i in range(len(x)):
    h = hash_coords(x[i], y[i], z[i])
    cell_start[h] += 1

# Step 3: Prefix sum to determine cell start indices
start = 0
for i in range(table_size):
    start += cell_start[i]
    cell_start[i] = start
cell_start[table_size] = start  # guard

# Step 4: Fill in object indices into cell entries
for i in range(num_objects):
    h = hash_coords(x[i], y[i], z[i])
    cell_start[h] -= 1
    cell_entries[cell_start[h]] = i

print(cell_entries)


def query(x, y, z, maxDist):
    x0 = int_coord(x - maxDist)
    y0 = int_coord(y - maxDist)
    z0 = int_coord(z - maxDist)

    x1 = int_coord(x + maxDist)
    y1 = int_coord(y + maxDist)
    z1 = int_coord(z + maxDist)

    query_ids = []
    query_size = 0

    for xi in range(x0, x1 + 1):
        for yi in range(y0, y1 + 1):
            for zi in range(z0, z1 + 1):
                h = hash_coords(xi, yi, zi)
                start = cell_start[h]
                end = cell_start[h + 1]

                for i in range(start, end):
                    query_ids.append(cell_entries[i])
                    query_size += 1

    return query_size, query_ids


query_size, query_ids = query(0.5, 0.5, 0.5, 2. * spacing)
print(query_size)
print(query_ids)
