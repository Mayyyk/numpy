# %% [markdown]
"""
# NUMPY COMPLETE COURSE - FROM ZERO TO HERO
## Based on official NumPy documentation + SciPy integration

**Course Structure:**
- Part 1: NumPy Essentials (2-3 days)
- Part 2: NumPy Intermediate (2-3 days)  
- Part 3: NumPy Advanced (2-3 days)
- Part 4: Data Science Integration - Pandas, Matplotlib, SciPy (2-3 days)
- Part 5: Real-World Projects (2-3 days)

**Total: ~2 weeks**

**How to use:**
1. Run each cell sequentially
2. Read explanations
3. Try exercises WITHOUT looking at solutions
4. Solutions are in collapsed cells or comments
"""

# %% [markdown]
"""
# Setup and Imports
Run this cell first to import all necessary libraries
"""

# %%
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

print("NumPy version:", np.__version__)
print("Setup complete! ✓")

# %% [markdown]
"""
---
# PART 1: NUMPY ESSENTIALS - THE FOUNDATION
---
"""

# %% [markdown]
"""
## 1.1 Array Creation - The Building Blocks

**Key Concept:** NumPy array (ndarray) = homogeneous, fixed-size, multi-dimensional

**Differences from Python lists:**
- Homogeneous (all same type) vs heterogeneous
- Fixed size vs dynamic
- Vectorized operations vs loops
- Much faster and more memory efficient
"""

# %%
# From lists
a = np.array([1, 2, 3, 4, 5])
print(f"1D array: {a}")

# 2D array
b = np.array([[1, 2, 3],
              [4, 5, 6]])
print(f"2D array:\n{b}")

# %%
# Specialized creation functions
zeros = np.zeros((3, 4))          # All zeros
ones = np.ones((2, 3))            # All ones
full = np.full((2, 2), 7)         # All same value
identity = np.eye(3)              # Identity matrix
arange = np.arange(0, 10, 2)      # Like range(): start, stop, step
linspace = np.linspace(0, 1, 5)   # N evenly spaced numbers

print("zeros:\n", zeros)
print("\nlinspace:", linspace)

# %%
# Random arrays - MODERN WAY (NumPy 1.17+)
rng = np.random.default_rng(42)   # Seed for reproducibility

random_uniform = rng.random((2, 3))           # Uniform [0, 1)
random_normal = rng.standard_normal((2, 3))   # Normal (mean=0, std=1)
random_int = rng.integers(0, 10, size=(3, 3)) # Random integers

print("Random uniform:\n", random_uniform)
print("\nRandom normal:\n", random_normal)

# %% [markdown]
"""
### EXERCISES 1.1 - Array Creation
Try to solve these without looking at solutions!
"""

# %%
# Exercise 1.1.1: Create array of even numbers from 0 to 20 (inclusive)
ex1_1_1 = None
# Your code here

# Exercise 1.1.2: Create 5x5 array with values 1-25 (hint: arange + reshape)
ex1_1_2 = None
# Your code here

# Exercise 1.1.3: Create array of 10 numbers evenly spaced between 0 and π
ex1_1_3 = None
# Your code here

# Exercise 1.1.4: Create 3x3 array with random integers between 1 and 100
ex1_1_4 = None
# Your code here

# Exercise 1.1.5: Create checkerboard pattern (8x8, alternating 0 and 1)
ex1_1_5 = None
# Your code here

# %% [markdown]
"""
<details>
<summary>Click to see solutions</summary>

```python
# Solution 1.1.1
ex1_1_1 = np.arange(0, 21, 2)

# Solution 1.1.2
ex1_1_2 = np.arange(1, 26).reshape(5, 5)

# Solution 1.1.3
ex1_1_3 = np.linspace(0, np.pi, 10)

# Solution 1.1.4
rng = np.random.default_rng()
ex1_1_4 = rng.integers(1, 101, size=(3, 3))

# Solution 1.1.5
ex1_1_5 = np.fromfunction(lambda i, j: (i + j) % 2, (8, 8), dtype=int)
```
</details>
"""

# %% [markdown]
"""
## 1.2 Array Attributes - Understanding Your Data

**Key Concept:** Every array has metadata describing its structure
"""

# %%
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8]])

print(f"Array:\n{arr}\n")
print(f"Shape: {arr.shape}")        # Dimensions (rows, cols)
print(f"Size: {arr.size}")          # Total elements
print(f"Ndim: {arr.ndim}")          # Number of dimensions
print(f"Dtype: {arr.dtype}")        # Data type
print(f"Itemsize: {arr.itemsize}")  # Bytes per element
print(f"Nbytes: {arr.nbytes}")      # Total bytes

# %%
# Data types
int_arr = np.array([1, 2, 3], dtype=np.int32)      # 32-bit integer
float_arr = np.array([1.0, 2.0], dtype=np.float64) # 64-bit float
bool_arr = np.array([True, False, True])           # Boolean

print(f"int32 uses {int_arr.itemsize} bytes per element")
print(f"float64 uses {float_arr.itemsize} bytes per element")

# Type conversion
arr_float = np.array([1.7, 2.3, 3.9])
arr_int = arr_float.astype(np.int32)  # Truncates
print(f"\nOriginal (float): {arr_float}")
print(f"Converted (int): {arr_int}")

# %% [markdown]
"""
### EXERCISES 1.2 - Array Attributes
"""

# %%
# Exercise 1.2.1: Create 3D array of shape (2, 4, 3) and print all attributes
ex1_2_1 = None
# Your code here

# Exercise 1.2.2: Create array with smallest integer type that can hold 0-100
ex1_2_2 = None
# Your code here

# Exercise 1.2.3: Calculate memory saved converting 1M float64 to float32
ex1_2_3_saved_mb = None
# Your code here

# %% [markdown]
"""
<details>
<summary>Solutions 1.2</summary>

```python
# Solution 1.2.1
ex1_2_1 = np.random.rand(2, 4, 3)
print(f"Shape: {ex1_2_1.shape}, Ndim: {ex1_2_1.ndim}")
print(f"Dtype: {ex1_2_1.dtype}, Nbytes: {ex1_2_1.nbytes}")

# Solution 1.2.2
ex1_2_2 = np.arange(101, dtype=np.uint8)  # uint8 = 0 to 255

# Solution 1.2.3
n = 1_000_000
original = n * 8  # float64
converted = n * 4  # float32
ex1_2_3_saved_mb = (original - converted) / 1024 / 1024
print(f"Saved: {ex1_2_3_saved_mb:.2f} MB")
```
</details>
"""

# %% [markdown]
"""
## 1.3 Indexing and Slicing - Accessing Your Data

**Key patterns:**
- `arr[i]` - single element or row
- `arr[i, j]` - element at row i, column j
- `arr[start:stop:step]` - slicing
- `arr[:, j]` - entire column
- `arr[i, :]` - entire row
"""

# %%
# 1D indexing
arr = np.array([10, 20, 30, 40, 50])
print(f"Array: {arr}")
print(f"First: {arr[0]}")
print(f"Last: {arr[-1]}")
print(f"Middle three: {arr[1:4]}")
print(f"Every other: {arr[::2]}")
print(f"Reversed: {arr[::-1]}")

# %%
# 2D indexing
arr2d = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

print(f"2D Array:\n{arr2d}\n")
print(f"Element [1,2]: {arr2d[1, 2]}")
print(f"First row: {arr2d[0, :]}")
print(f"Second column: {arr2d[:, 1]}")
print(f"Top-left 2x2:\n{arr2d[:2, :2]}")

# %%
# Boolean indexing - POWERFUL!
arr = np.array([1, 5, 3, 8, 2, 9, 4])
mask = arr > 4
print(f"Array: {arr}")
print(f"Mask (>4): {mask}")
print(f"Elements >4: {arr[mask]}")

# Fancy indexing
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([0, 2, 4])
print(f"\nElements at [0,2,4]: {arr[indices]}")

# %%
# FLAT indexing - treats array as 1D
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])
print(f"2D array:\n{arr2d}")
print(f"Flat index 4: {arr2d.flat[4]}")  # Element 5

arr2d.flat[::2] = 0  # Every other element
print(f"After flat[::2] = 0:\n{arr2d}")

# %% [markdown]
"""
### EXERCISES 1.3 - Indexing and Slicing
"""

# %%
# Exercise 1.3.1: Extract [30, 60, 90] from array
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
ex1_3_1 = None
# Your code here

# Exercise 1.3.2: Extract middle 2x2 subarray
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]])
ex1_3_2 = None
# Expected: [[6, 7], [10, 11]]

# Exercise 1.3.3: Get all even numbers
arr = np.array([15, 22, 7, 34, 9, 12, 45, 18])
ex1_3_3 = None

# Exercise 1.3.4: Get diagonal elements (where row == col)
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
ex1_3_4 = None
# Expected: [1, 5, 9]

# Exercise 1.3.5: Replace all negative values with 0
arr = np.array([-1, 5, -3, 8, -2, 7])
ex1_3_5 = None

# %% [markdown]
"""
<details>
<summary>Solutions 1.3</summary>

```python
# Solution 1.3.1
ex1_3_1 = arr[2::3]  # Start at 2, step 3

# Solution 1.3.2
ex1_3_2 = arr[1:3, 1:3]

# Solution 1.3.3
ex1_3_3 = arr[arr % 2 == 0]

# Solution 1.3.4
ex1_3_4 = np.diag(arr)

# Solution 1.3.5
ex1_3_5 = arr.copy()
ex1_3_5[ex1_3_5 < 0] = 0
```
</details>
"""

# %% [markdown]
"""
## 1.4 Array Operations - Computing with Arrays

**Key Concept:** Vectorized operations = element-wise operations without loops

**Categories:**
1. Arithmetic: +, -, *, /, **, %
2. Comparison: <, >, ==, !=
3. Logical: &, |, ~
4. Universal functions: np.sin, np.exp, np.sqrt
"""

# %%
# Arithmetic (element-wise!)
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(f"a = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")  # NOT dot product!
print(f"a ** 2 = {a ** 2}")
print(f"b / 10 = {b / 10}")

# %%
# Comparison and logical operations
print(f"a > 2: {a > 2}")
print(f"(a > 1) & (a < 4): {(a > 1) & (a < 4)}")  # AND
print(f"(a < 2) | (a > 3): {(a < 2) | (a > 3)}")  # OR

# %%
# Universal functions (ufuncs)
arr = np.array([0, np.pi/2, np.pi])
print(f"arr = {arr}")
print(f"np.sin(arr) = {np.sin(arr)}")
print(f"np.exp(arr) = {np.exp(arr)}")

arr = np.array([1, 4, 9, 16])
print(f"\narr = {arr}")
print(f"np.sqrt(arr) = {np.sqrt(arr)}")

# %%
# Aggregations
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Sum: {arr.sum()}")
print(f"Mean: {arr.mean()}")
print(f"Max: {arr.max()}")
print(f"Std: {arr.std()}")

# %%
# Axis-wise operations - IMPORTANT!
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])

print(f"Array:\n{arr2d}\n")
print(f"Sum all: {arr2d.sum()}")
print(f"Sum axis=0 (columns): {arr2d.sum(axis=0)}")
print(f"Sum axis=1 (rows): {arr2d.sum(axis=1)}")
print(f"Mean per column: {arr2d.mean(axis=0)}")
print(f"Max per row: {arr2d.max(axis=1)}")

# %% [markdown]
"""
### EXERCISES 1.4 - Array Operations
"""

# %%
# Exercise 1.4.1: Convert Celsius to Fahrenheit (F = C * 9/5 + 32)
celsius = np.array([0, 10, 20, 30, 40])
fahrenheit = None

# Exercise 1.4.2: Find elements divisible by both 2 AND 3
arr = np.array([6, 9, 12, 15, 18, 21, 24, 27, 30])
ex1_4_2 = None

# Exercise 1.4.3: Normalize array (mean=0, std=1)
arr = np.array([10, 20, 30, 40, 50])
ex1_4_3 = None

# Exercise 1.4.4: Calculate distances from origin for 2D points
points = np.array([[0, 0], [3, 4], [1, 1], [5, 12]])
distances = None
# Hint: distance = sqrt(x^2 + y^2)

# Exercise 1.4.5: Find column with highest mean
arr = np.array([[1, 5, 3],
                [2, 8, 4],
                [3, 6, 5]])
ex1_4_5 = None  # Should return column index

# %% [markdown]
"""
<details>
<summary>Solutions 1.4</summary>

```python
# Solution 1.4.1
fahrenheit = celsius * 9/5 + 32

# Solution 1.4.2
ex1_4_2 = arr[(arr % 2 == 0) & (arr % 3 == 0)]

# Solution 1.4.3
ex1_4_3 = (arr - arr.mean()) / arr.std()

# Solution 1.4.4
distances = np.sqrt((points ** 2).sum(axis=1))

# Solution 1.4.5
ex1_4_5 = arr.mean(axis=0).argmax()
```
</details>
"""

# %% [markdown]
"""
---
# PART 2: NUMPY INTERMEDIATE - POWER FEATURES
---
"""

# %% [markdown]
"""
## 2.1 Broadcasting - Automatic Array Expansion

**Key Concept:** Broadcasting = automatic expansion of smaller arrays

**Rules:**
1. If different ndim, prepend 1s to smaller shape
2. Arrays compatible if dimensions equal OR one is 1
3. Broadcast if compatible in all dimensions

**Examples:**
- (3, 4) + (3, 4) → OK
- (3, 4) + (1, 4) → OK
- (3, 4) + (4,) → OK
- (3, 4) + (3, 5) → ERROR
"""

# %%
# Example 1: Scalar + Array
arr = np.array([1, 2, 3])
result = arr + 10  # 10 broadcasts to [10, 10, 10]
print(f"[1,2,3] + 10 = {result}")

# %%
# Example 2: 1D + 2D
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
row = np.array([10, 20, 30])
result = matrix + row  # row broadcast to each row
print(f"Matrix:\n{matrix}")
print(f"+ row {row}")
print(f"Result:\n{result}")

# %%
# Example 3: Column + Row = Matrix
col = np.array([[1], [2], [3]])  # (3, 1)
row = np.array([10, 20, 30])     # (3,) → (1, 3)
result = col + row               # Broadcasts to (3, 3)
print(f"Column:\n{col}")
print(f"+ Row: {row}")
print(f"= Matrix:\n{result}")

# %%
# Practical: Normalize each column
data = np.array([[1, 4, 7],
                 [2, 5, 8],
                 [3, 6, 9]], dtype=float)

col_means = data.mean(axis=0)  # (3,)
normalized = data - col_means   # Broadcasting!

print(f"Original:\n{data}")
print(f"Column means: {col_means}")
print(f"Normalized:\n{normalized}")

# %% [markdown]
"""
### EXERCISES 2.1 - Broadcasting
"""

# %%
# Exercise 2.1.1: Add [1, 2, 3] to each row of 4x3 matrix of ones
ex2_1_1 = None

# Exercise 2.1.2: Multiply each column by [1, 2, 3] for 3x3 matrix
ex2_1_2 = None

# Exercise 2.1.3: Create 5x5 multiplication table using broadcasting
ex2_1_3 = None
# [[1,2,3,4,5], [2,4,6,8,10], ..., [5,10,15,20,25]]

# Exercise 2.1.4: Normalize each row (subtract row mean)
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]], dtype=float)
ex2_1_4 = None

# Exercise 2.1.5: Distance matrix for 1D points [0, 1, 3, 6]
points = np.array([0, 1, 3, 6])
ex2_1_5 = None

# %% [markdown]
"""
<details>
<summary>Solutions 2.1</summary>

```python
# Solution 2.1.1
matrix = np.ones((4, 3))
row = np.array([1, 2, 3])
ex2_1_1 = matrix + row

# Solution 2.1.2
matrix = np.ones((3, 3))
col = np.array([[1], [2], [3]])
ex2_1_2 = matrix * col

# Solution 2.1.3
row = np.arange(1, 6)
col = np.arange(1, 6).reshape(-1, 1)
ex2_1_3 = row * col

# Solution 2.1.4
row_means = arr.mean(axis=1, keepdims=True)
ex2_1_4 = arr - row_means

# Solution 2.1.5
ex2_1_5 = np.abs(points[:, np.newaxis] - points)
```
</details>
"""

# %% [markdown]
"""
## 2.2 Reshaping and Transposing

**Key operations:**
- `reshape`: change dimensions
- `flatten/ravel`: to 1D
- `transpose/.T`: swap axes
- `squeeze/expand_dims`: remove/add dimensions
"""

# %%
# Reshape
arr = np.arange(12)
print(f"Original (12,): {arr}")

reshaped = arr.reshape(3, 4)
print(f"\nReshaped (3, 4):\n{reshaped}")

reshaped = arr.reshape(2, 3, 2)
print(f"\nReshaped (2, 3, 2):\n{reshaped}")

# %%
# -1 means "figure it out"
arr = np.arange(24)
reshaped = arr.reshape(4, -1)  # -1 becomes 6
print(f"Reshape (4, -1):\n{reshaped}")

# %%
# Flatten vs ravel
arr2d = np.array([[1, 2, 3], [4, 5, 6]])
flat = arr2d.flatten()  # Always returns copy
ravel = arr2d.ravel()   # Returns view if possible

print(f"Original:\n{arr2d}")
print(f"Flattened: {flat}")
print(f"Raveled: {ravel}")

# %%
# Transpose
arr = np.array([[1, 2, 3],
                [4, 5, 6]])
print(f"Original (2, 3):\n{arr}")
print(f"Transposed (3, 2):\n{arr.T}")

# %%
# Expand/squeeze dimensions
arr = np.array([1, 2, 3])
print(f"Original: {arr.shape}")

expanded = np.expand_dims(arr, axis=0)
print(f"Expanded axis=0: {expanded.shape}")  # (1, 3)

expanded = np.expand_dims(arr, axis=1)
print(f"Expanded axis=1: {expanded.shape}")  # (3, 1)

# %% [markdown]
"""
### EXERCISES 2.2 - Reshaping
"""

# %%
# Exercise 2.2.1: Create 1-16, reshape to (4,4), then (2,8)
ex2_2_1a = None
ex2_2_1b = None

# Exercise 2.2.2: Flatten 3D to 1D, reshape to (6, -1)
arr3d = np.random.rand(2, 3, 4)
ex2_2_2 = None

# Exercise 2.2.3: Swap first and last dimensions of (2,3,4,5) array
arr = np.random.rand(2, 3, 4, 5)
ex2_2_3 = None  # Should be (5,3,4,2)

# Exercise 2.2.4: Create column vector from [1,2,3,4,5]
arr = np.array([1, 2, 3, 4, 5])
ex2_2_4 = None  # Shape (5, 1)

# %% [markdown]
"""
<details>
<summary>Solutions 2.2</summary>

```python
# Solution 2.2.1
ex2_2_1a = np.arange(1, 17).reshape(4, 4)
ex2_2_1b = np.arange(1, 17).reshape(2, 8)

# Solution 2.2.2
flat = arr3d.flatten()
ex2_2_2 = flat.reshape(6, -1)

# Solution 2.2.3
ex2_2_3 = arr.transpose(3, 1, 2, 0)

# Solution 2.2.4
ex2_2_4 = arr.reshape(-1, 1)
```
</details>
"""

# %% [markdown]
"""
## 2.3 Stacking and Splitting

**Stacking:**
- `np.vstack`: vertical (rows)
- `np.hstack`: horizontal (columns)
- `np.concatenate`: general
- `np.stack`: create new axis

**Splitting:**
- `np.vsplit/hsplit/split`
"""

# %%
# Stacking
a = np.array([[1, 2, 3]])
b = np.array([[4, 5, 6]])

vstacked = np.vstack([a, b])
print("Vertical stack:\n", vstacked)

hstacked = np.hstack([a.T, b.T])
print("\nHorizontal stack:\n", hstacked)

# %%
# Concatenate
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

concat_v = np.concatenate([a, b], axis=0)
concat_h = np.concatenate([a, b], axis=1)

print("Concat axis=0:\n", concat_v)
print("\nConcat axis=1:\n", concat_h)

# %%
# Splitting
arr = np.array([[1, 2, 3, 4, 5, 6],
                [7, 8, 9, 10, 11, 12]])

split_h = np.hsplit(arr, 3)  # 3 equal parts
print("Horizontal split into 3:")
for i, s in enumerate(split_h):
    print(f"Part {i}:\n{s}\n")

# %% [markdown]
"""
### EXERCISES 2.3 - Stacking
"""

# %%
# Exercise 2.3.1: Stack [1,2], [3,4], [5,6] vertically
a = np.array([1, 2])
b = np.array([3, 4])
c = np.array([5, 6])
ex2_3_1 = None

# Exercise 2.3.2: Create 4x4 by stacking 4 2x2 matrices
# TL:[[1,2],[3,4]], TR:[[5,6],[7,8]]
# BL:[[9,10],[11,12]], BR:[[13,14],[15,16]]
ex2_3_2 = None

# %% [markdown]
"""
## 2.4 Copying and Views

**Key difference:**
- View: Same data, different object
- Copy: Different data, different object

**Slicing creates views!**
"""

# %%
# Views from slicing
arr = np.array([1, 2, 3, 4, 5])
view = arr[1:4]  # View!

print(f"Original: {arr}")
print(f"View: {view}")

view[0] = 999  # Modifies original!
print(f"\nAfter view[0] = 999:")
print(f"Original: {arr}")  # Changed!

# %%
# Explicit copy
arr = np.array([1, 2, 3, 4, 5])
copy = arr[1:4].copy()  # Copy

copy[0] = 999
print(f"Original: {arr}")  # Unchanged
print(f"Copy: {copy}")

# %%
# Check if view or copy
arr = np.array([1, 2, 3])
view = arr[:]
copy = arr.copy()

print(f"view.base is arr: {view.base is arr}")  # True
print(f"copy.base is arr: {copy.base is arr}")  # False

# %% [markdown]
"""
---
# PART 3: NUMPY ADVANCED - MASTERY
---
"""

# %% [markdown]
"""
## 3.1 Advanced Indexing

**Techniques:**
1. Integer array indexing
2. Boolean indexing
3. `np.where` conditional selection
4. `argmax/argmin` for finding indices
"""

# %%
# Integer array indexing
arr = np.array([10, 20, 30, 40, 50])
indices = np.array([0, 2, 4])
selected = arr[indices]
print(f"Array: {arr}")
print(f"Selected [0,2,4]: {selected}")

# %%
# 2D integer indexing - get diagonal
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
rows = np.array([0, 1, 2])
cols = np.array([0, 1, 2])
diagonal = arr2d[rows, cols]
print(f"Array:\n{arr2d}")
print(f"Diagonal: {diagonal}")

# %%
# np.where - conditional selection
arr = np.array([1, 2, 3, 4, 5, 6])
result = np.where(arr > 3, arr * 2, arr)
print(f"Original: {arr}")
print(f"If >3 double else keep: {result}")

# Find indices
indices = np.where(arr > 3)
print(f"Indices where >3: {indices}")

# %%
# argmax/argmin
arr = np.array([3, 7, 1, 9, 2])
print(f"Array: {arr}")
print(f"Index of max: {arr.argmax()}")
print(f"Index of min: {arr.argmin()}")

arr2d = np.array([[1, 5, 3],
                  [9, 2, 7]])
print(f"\n2D array:\n{arr2d}")
print(f"Argmax per row: {arr2d.argmax(axis=1)}")
print(f"Argmax per col: {arr2d.argmax(axis=0)}")

# %% [markdown]
"""
### EXERCISES 3.1 - Advanced Indexing
"""

# %%
# Exercise 3.1.1: Select elements at positions [0,3,6,9]
arr = np.arange(20)
ex3_1_1 = None

# Exercise 3.1.2: Clip values to [10, 50] range
arr = np.array([5, 25, 75, 100, 3, 45, 60])
ex3_1_2 = None

# Exercise 3.1.3: Find indices of local maxima
arr = np.array([1, 3, 7, 1, 2, 6, 0, 1])
ex3_1_3 = None

# %% [markdown]
"""
## 3.2 Linear Algebra

**Key operations:**
- Dot/matrix product: `@` or `np.dot`
- Transpose: `.T`
- Inverse: `np.linalg.inv`
- Determinant: `np.linalg.det`
- Eigenvalues: `np.linalg.eig`
- Solve systems: `np.linalg.solve`
"""

# %%
# Dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
dot = np.dot(a, b)
print(f"Dot product: {dot}")

# %%
# Matrix multiplication
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = A @ B  # Preferred for matrices
print(f"A @ B:\n{C}")
print(f"\nA * B (element-wise):\n{A * B}")

# %%
# Inverse and determinant
A = np.array([[1, 2], [3, 4]], dtype=float)
A_inv = np.linalg.inv(A)
det = np.linalg.det(A)

print(f"A:\n{A}")
print(f"\nA inverse:\n{A_inv}")
print(f"\nDeterminant: {det}")
print(f"\nA @ A_inv:\n{A @ A_inv}")

# %%
# Solving Ax = b
A = np.array([[3, 1], [1, 2]], dtype=float)
b = np.array([9, 8])
x = np.linalg.solve(A, b)

print(f"A:\n{A}")
print(f"b: {b}")
print(f"x: {x}")
print(f"Verify A@x: {A @ x}")

# %% [markdown]
"""
### EXERCISES 3.2 - Linear Algebra
"""

# %%
# Exercise 3.2.1: Calculate Euclidean norm of [3, 4]
v = np.array([3, 4])
ex3_2_1 = None  # Should be 5.0

# Exercise 3.2.2: Check if matrix is symmetric
A = np.array([[1, 2, 3], [2, 4, 5], [3, 5, 6]])
ex3_2_2 = None  # Boolean

# Exercise 3.2.3: Project b onto a: proj = (a·b / a·a) * a
a = np.array([1, 0, 0])
b = np.array([1, 1, 0])
ex3_2_3 = None

# Exercise 3.2.4: Angle between vectors (radians)
v1 = np.array([1, 0])
v2 = np.array([1, 1])
ex3_2_4 = None

# %% [markdown]
"""
## 3.3 Random Number Generation

**Modern way:** `rng = np.random.default_rng(seed)`

**Distributions:**
- Uniform: `rng.random()`
- Normal: `rng.normal()` / `rng.standard_normal()`
- Integers: `rng.integers()`
- Choice: `rng.choice()`
- Permutation: `rng.permutation()`
"""

# %%
rng = np.random.default_rng(42)

# Distributions
uniform = rng.random((3, 3))
normal = rng.standard_normal((3, 3))
integers = rng.integers(0, 10, size=(3, 3))

print("Uniform [0,1):\n", uniform)
print("\nNormal (μ=0, σ=1):\n", normal)

# %%
# Custom normal
custom = rng.normal(loc=10, scale=2, size=1000)
print(f"Custom normal (μ=10, σ=2):")
print(f"Mean: {custom.mean():.2f}")
print(f"Std: {custom.std():.2f}")

# %%
# Random choice and permutation
arr = np.array([10, 20, 30, 40, 50])
choice = rng.choice(arr, size=3, replace=False)
shuffled = rng.permutation(arr)

print(f"Original: {arr}")
print(f"Choice (no replacement): {choice}")
print(f"Shuffled: {shuffled}")

# %% [markdown]
"""
## 3.4 Performance and Vectorization

**Golden Rule:** If you write a `for` loop over array elements, you're doing it wrong!

**Why NumPy is fast:**
1. Vectorization - operations in compiled C
2. Broadcasting - avoid loops
3. Views - avoid unnecessary copies
"""

# %%
# Performance comparison
n = 1_000_000

# Python list
python_list = list(range(n))
start = time.time()
result = [x ** 2 for x in python_list]
time_list = time.time() - start

# NumPy vectorized
numpy_array = np.arange(n)
start = time.time()
result = numpy_array ** 2
time_numpy = time.time() - start

print(f"Python list: {time_list:.4f}s")
print(f"NumPy array: {time_numpy:.4f}s")
print(f"Speedup: {time_list / time_numpy:.0f}x faster!")

# %%
# Bad vs Good practice
arr = np.random.rand(10000)

# BAD - loop
start = time.time()
result = np.empty_like(arr)
for i in range(len(arr)):
    result[i] = np.sin(arr[i])
time_loop = time.time() - start

# GOOD - vectorized
start = time.time()
result = np.sin(arr)
time_vectorized = time.time() - start

print(f"With loop: {time_loop:.4f}s")
print(f"Vectorized: {time_vectorized:.4f}s")
print(f"Speedup: {time_loop / time_vectorized:.0f}x")

# %% [markdown]
"""
### EXERCISES 3.4 - Vectorization
"""

# %%
# Exercise 3.4.1: Vectorize: clip to [0, 1]
arr = np.random.randn(1000)
ex3_4_1 = None  # Single line using np.clip

# Exercise 3.4.2: Pairwise distances using broadcasting
points = np.random.rand(10, 2)
ex3_4_2 = None  # No nested loops!

# %% [markdown]
"""
---
# PART 4: DATA SCIENCE INTEGRATION
## Pandas, Matplotlib, SciPy
---
"""

# %% [markdown]
"""
## 4.1 Pandas Basics - DataFrames

**Pandas = NumPy + labels + missing data**

Install: `pip install pandas`
"""

# %%
try:
    import pandas as pd
    print("Pandas version:", pd.__version__)
    
    # Create DataFrame from NumPy
    data = np.random.randn(5, 3)
    df = pd.DataFrame(data, columns=['A', 'B', 'C'])
    print("\nDataFrame from NumPy:\n", df)
    
except ImportError:
    print("Install pandas: pip install pandas")

# %%
# DataFrame from dictionary
data_dict = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 28],
    'score': [85.5, 92.0, 78.5, 88.0]
}
df = pd.DataFrame(data_dict)
print("DataFrame:\n", df)

# %%
# Access and filter
print("Ages:\n", df['age'])
print("\nFirst row:\n", df.iloc[0])
print("\nScore > 85:\n", df[df['score'] > 85])

# %%
# Add column (vectorized!)
df['score_squared'] = df['score'] ** 2
print("With new column:\n", df)

# %%
# Statistics
print("Statistics:\n", df[['age', 'score']].describe())

# %%
# GroupBy
df['group'] = ['A', 'B', 'A', 'B']
grouped = df.groupby('group')['score'].mean()
print("Mean score by group:\n", grouped)

# %%
# Convert to NumPy
values = df[['age', 'score']].values
print("As NumPy array:\n", values)
print("Type:", type(values))

# %% [markdown]
"""
## 4.2 Matplotlib Basics - Visualization

**Matplotlib = plotting library**

Install: `pip install matplotlib`
"""

# %%
try:
    import matplotlib.pyplot as plt
    print("Matplotlib imported successfully!")
    
    # Line plot
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x, y)
    plt.title('Sine Wave')
    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.grid(True)
    
    # Scatter plot
    plt.subplot(1, 2, 2)
    x = np.random.randn(100)
    y = 2 * x + np.random.randn(100) * 0.5
    plt.scatter(x, y, alpha=0.5)
    plt.title('Scatter Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("Install matplotlib: pip install matplotlib")

# %%
# Histogram
data = np.random.normal(0, 1, 1000)
plt.figure(figsize=(8, 5))
plt.hist(data, bins=30, alpha=0.7, edgecolor='black')
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

# %%
# Heatmap
data = np.random.rand(10, 10)
plt.figure(figsize=(8, 6))
plt.imshow(data, cmap='hot', interpolation='nearest')
plt.colorbar(label='Value')
plt.title('Heatmap')
plt.show()

# %%
# Multiple plots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot 1: Line
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('Sine')

# Plot 2: Scatter
axes[0, 1].scatter(np.random.randn(50), np.random.randn(50))
axes[0, 1].set_title('Scatter')

# Plot 3: Bar
axes[1, 0].bar(['A', 'B', 'C'], [3, 7, 5])
axes[1, 0].set_title('Bar Chart')

# Plot 4: Histogram
axes[1, 1].hist(np.random.randn(1000), bins=20)
axes[1, 1].set_title('Histogram')

plt.tight_layout()
plt.show()

# %% [markdown]
"""
## 4.3 SciPy - Scientific Computing

**SciPy = Advanced scientific algorithms**

**Modules:**
- `scipy.optimize` - optimization, curve fitting
- `scipy.interpolate` - interpolation
- `scipy.signal` - signal processing, FFT
- `scipy.stats` - statistical functions
- `scipy.integrate` - integration
- `scipy.spatial` - spatial algorithms

Install: `pip install scipy`
"""

# %%
try:
    import scipy
    print("SciPy version:", scipy.__version__)
    from scipy import optimize, interpolate, signal, stats, integrate
    
except ImportError:
    print("Install scipy: pip install scipy")

# %% [markdown]
"""
### 4.3.1 SciPy Optimization
"""

# %%
from scipy import optimize

# Minimize function: f(x) = x^2 + 10*sin(x)
def f(x):
    return x**2 + 10*np.sin(x)

# Find minimum
result = optimize.minimize(f, x0=0)
print("Minimum found at x =", result.x[0])
print("Function value:", result.fun)

# Plot to visualize
x = np.linspace(-5, 5, 200)
plt.figure(figsize=(10, 4))
plt.plot(x, f(x), label='f(x)')
plt.plot(result.x, result.fun, 'ro', markersize=10, label='Minimum')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function Minimization')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Curve fitting
from scipy.optimize import curve_fit

# Generate noisy data
x_data = np.linspace(0, 4, 50)
y_data = 2.5 * np.exp(-x_data / 1.5) + np.random.normal(0, 0.2, 50)

# Define model
def exponential(x, a, b):
    return a * np.exp(-x / b)

# Fit curve
params, covariance = curve_fit(exponential, x_data, y_data)
print(f"Fitted parameters: a={params[0]:.2f}, b={params[1]:.2f}")

# Plot
plt.figure(figsize=(10, 4))
plt.scatter(x_data, y_data, alpha=0.5, label='Data')
plt.plot(x_data, exponential(x_data, *params), 'r-', label='Fit')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Curve Fitting')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
"""
### 4.3.2 SciPy Interpolation
"""

# %%
from scipy import interpolate

# Original sparse data
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 1, 4, 9, 16, 25])

# Create interpolation function
f_linear = interpolate.interp1d(x, y, kind='linear')
f_cubic = interpolate.interp1d(x, y, kind='cubic')

# Interpolate at new points
x_new = np.linspace(0, 5, 100)
y_linear = f_linear(x_new)
y_cubic = f_cubic(x_new)

# Plot
plt.figure(figsize=(10, 4))
plt.plot(x, y, 'o', label='Original data')
plt.plot(x_new, y_linear, '--', label='Linear interpolation')
plt.plot(x_new, y_cubic, '-', label='Cubic interpolation')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Interpolation')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
"""
### 4.3.3 SciPy Signal Processing
"""

# %%
from scipy import signal

# Generate noisy signal
t = np.linspace(0, 1, 1000)
clean_signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine
noise = np.random.normal(0, 0.5, 1000)
noisy_signal = clean_signal + noise

# Apply low-pass filter
b, a = signal.butter(4, 0.1)  # 4th order Butterworth filter
filtered_signal = signal.filtfilt(b, a, noisy_signal)

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(t[:200], clean_signal[:200])
plt.title('Clean Signal')
plt.xlabel('Time')

plt.subplot(1, 3, 2)
plt.plot(t[:200], noisy_signal[:200])
plt.title('Noisy Signal')
plt.xlabel('Time')

plt.subplot(1, 3, 3)
plt.plot(t[:200], filtered_signal[:200])
plt.title('Filtered Signal')
plt.xlabel('Time')

plt.tight_layout()
plt.show()

# %%
# FFT - Frequency analysis
from scipy.fft import fft, fftfreq

# Signal with multiple frequencies
t = np.linspace(0, 2, 1000)
signal_mixed = (np.sin(2 * np.pi * 5 * t) + 
                0.5 * np.sin(2 * np.pi * 10 * t))

# Compute FFT
fft_values = fft(signal_mixed)
fft_freq = fftfreq(len(t), t[1] - t[0])

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t[:100], signal_mixed[:100])
plt.title('Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.subplot(1, 2, 2)
plt.plot(fft_freq[:len(fft_freq)//2], 
         np.abs(fft_values[:len(fft_values)//2]))
plt.title('Frequency Domain (FFT)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 20)

plt.tight_layout()
plt.show()

# %% [markdown]
"""
### 4.3.4 SciPy Statistics
"""

# %%
from scipy import stats

# Generate samples from different distributions
normal_data = np.random.normal(0, 1, 1000)
uniform_data = np.random.uniform(-2, 2, 1000)

# Statistical tests
print("Descriptive Statistics:")
print(f"Mean: {stats.describe(normal_data).mean:.3f}")
print(f"Variance: {stats.describe(normal_data).variance:.3f}")
print(f"Skewness: {stats.skew(normal_data):.3f}")
print(f"Kurtosis: {stats.kurtosis(normal_data):.3f}")

# %%
# T-test: compare two samples
sample1 = np.random.normal(0, 1, 100)
sample2 = np.random.normal(0.5, 1, 100)

t_stat, p_value = stats.ttest_ind(sample1, sample2)
print(f"\nT-test: t={t_stat:.3f}, p={p_value:.3f}")
if p_value < 0.05:
    print("Samples are significantly different!")
else:
    print("No significant difference")

# %%
# Correlation
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

correlation, p_value = stats.pearsonr(x, y)
print(f"Pearson correlation: r={correlation:.3f}, p={p_value:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Correlation: r={correlation:.3f}')
plt.grid(True)
plt.show()

# %%
# Distributions
x = np.linspace(-4, 4, 100)

# Different distributions
distributions = [
    ('Normal', stats.norm(0, 1)),
    ('t-distribution', stats.t(df=5)),
    ('Chi-square', stats.chi2(df=3))
]

plt.figure(figsize=(12, 4))
for i, (name, dist) in enumerate(distributions, 1):
    plt.subplot(1, 3, i)
    plt.plot(x, dist.pdf(x))
    plt.title(name)
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
"""
### 4.3.5 SciPy Integration
"""

# %%
from scipy import integrate

# Integrate function: ∫(x^2) dx from 0 to 1
def f(x):
    return x**2

result, error = integrate.quad(f, 0, 1)
print(f"∫x² dx from 0 to 1 = {result:.6f}")
print(f"Error estimate: {error:.2e}")
print(f"Analytical result: 1/3 = {1/3:.6f}")

# %%
# Solve ODE: dy/dt = -y, y(0) = 1
def dydt(y, t):
    return -y

t = np.linspace(0, 5, 100)
y0 = 1.0
solution = integrate.odeint(dydt, y0, t)

plt.figure(figsize=(10, 4))
plt.plot(t, solution, label='Numerical solution')
plt.plot(t, np.exp(-t), '--', label='Analytical: e^(-t)')
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.title('ODE Solution')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
"""
### 4.3.6 SciPy Spatial - Distance and Clustering
"""

# %%
from scipy.spatial import distance, KDTree

# Pairwise distances
points = np.random.rand(5, 2)

# Euclidean distance matrix
dist_matrix = distance.cdist(points, points, metric='euclidean')
print("Distance matrix:\n", dist_matrix)

plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], s=100)
for i, point in enumerate(points):
    plt.annotate(f'P{i}', point, fontsize=12)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points in 2D Space')
plt.grid(True)
plt.show()

# %%
# KDTree for fast nearest neighbor search
points = np.random.rand(1000, 2)
tree = KDTree(points)

# Find 5 nearest neighbors to origin
query_point = np.array([0.5, 0.5])
distances, indices = tree.query(query_point, k=5)

print(f"Query point: {query_point}")
print(f"5 nearest neighbors:")
for i, (dist, idx) in enumerate(zip(distances, indices)):
    print(f"  {i+1}. Point {idx} at distance {dist:.3f}")

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(points[:, 0], points[:, 1], alpha=0.3, s=10)
plt.scatter(query_point[0], query_point[1], c='red', s=100, 
            marker='*', label='Query point')
plt.scatter(points[indices, 0], points[indices, 1], c='green', 
            s=50, label='Nearest neighbors')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('KDTree Nearest Neighbor Search')
plt.legend()
plt.grid(True)
plt.show()

# %% [markdown]
"""
### EXERCISES 4.3 - SciPy Integration
"""

# %%
# Exercise 4.3.1: Fit exponential decay to data
t_data = np.array([0, 1, 2, 3, 4, 5])
y_data = np.array([10, 6.7, 4.5, 3.0, 2.0, 1.4])
# Fit: y = a * exp(-b * t)

# Exercise 4.3.2: Apply moving average filter to noisy signal
noisy = np.sin(np.linspace(0, 4*np.pi, 200)) + np.random.randn(200) * 0.3
# Use signal.savgol_filter with window=15, polyorder=3

# Exercise 4.3.3: Test if data is normally distributed
data = np.random.randn(1000)
# Use stats.normaltest

# Exercise 4.3.4: Find all points within radius 0.2 of origin
points = np.random.rand(100, 2)
# Use KDTree.query_ball_point

# %% [markdown]
"""
---
# PART 5: REAL-WORLD PROJECTS
---

**Choose one or more projects to apply everything:**

1. **Image Processing**
   - Load/manipulate images as arrays
   - Apply filters (blur, edge detection)
   - Color transformations

2. **Time Series Analysis**
   - Sensor data analysis
   - Moving averages
   - Anomaly detection
   - FFT frequency analysis

3. **Machine Learning Data Prep**
   - Load Kaggle dataset
   - Data cleaning
   - Feature engineering
   - Train/test split
   - Normalization

4. **Physics Simulation**
   - Projectile motion
   - Heat diffusion
   - N-body problem
   - Wave equation

5. **Robotics: IMU Data Analysis**
   - Load IMU sensor data
   - Signal filtering
   - Motion classification
   - Vibration analysis
"""

# %% [markdown]
"""
## PROJECT 1: Time Series Analysis - IMU Sensor Data

**Task:** Analyze accelerometer data to detect motion patterns

We'll simulate IMU data (or you can load real data from Kaggle)
"""

# %%
# Simulate IMU data (accelerometer x, y, z)
np.random.seed(42)
t = np.linspace(0, 10, 1000)

# Simulate different motion states
walking = np.sin(2 * np.pi * 2 * t) + np.random.randn(1000) * 0.2
stationary = np.random.randn(1000) * 0.1
running = 2 * np.sin(2 * np.pi * 4 * t) + np.random.randn(1000) * 0.3

# Combine: 3s stationary, 4s walking, 3s running
acc_x = np.concatenate([
    stationary[:300],
    walking[300:700],
    running[700:]
])

acc_y = np.roll(acc_x, 100) + np.random.randn(1000) * 0.1
acc_z = np.roll(acc_x, 200) + np.random.randn(1000) * 0.1

# Create DataFrame
df = pd.DataFrame({
    'time': t,
    'acc_x': acc_x,
    'acc_y': acc_y,
    'acc_z': acc_z
})

print("IMU Data:")
print(df.head())

# %%
# Visualize raw data
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['acc_x'], label='X', alpha=0.7)
plt.plot(df['time'], df['acc_y'], label='Y', alpha=0.7)
plt.plot(df['time'], df['acc_z'], label='Z', alpha=0.7)
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Raw IMU Data')
plt.legend()
plt.grid(True)
plt.show()

# %%
# Calculate magnitude
df['magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)

plt.figure(figsize=(12, 4))
plt.plot(df['time'], df['magnitude'])
plt.xlabel('Time (s)')
plt.ylabel('Acceleration Magnitude')
plt.title('Acceleration Magnitude Over Time')
plt.grid(True)
plt.show()

# %%
# Apply moving average filter
from scipy.ndimage import uniform_filter1d

window_size = 20
df['magnitude_filtered'] = uniform_filter1d(df['magnitude'], window_size)

plt.figure(figsize=(12, 4))
plt.plot(df['time'], df['magnitude'], alpha=0.3, label='Raw')
plt.plot(df['time'], df['magnitude_filtered'], label='Filtered', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Magnitude')
plt.title('Filtered Signal')
plt.legend()
plt.grid(True)
plt.show()

# %%
# FFT Analysis
from scipy.fft import fft, fftfreq

# FFT on each segment
segments = [
    (0, 300, 'Stationary'),
    (300, 700, 'Walking'),
    (700, 1000, 'Running')
]

plt.figure(figsize=(15, 4))
for i, (start, end, label) in enumerate(segments, 1):
    segment_data = df['magnitude'].iloc[start:end].values
    fft_values = np.abs(fft(segment_data))
    fft_freq = fftfreq(len(segment_data), t[1] - t[0])
    
    plt.subplot(1, 3, i)
    plt.plot(fft_freq[:len(fft_freq)//2], fft_values[:len(fft_values)//2])
    plt.title(f'FFT: {label}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(0, 10)
    plt.grid(True)

plt.tight_layout()
plt.show()

# %%
# Feature extraction for classification
def extract_features(signal, window_size=100):
    """Extract statistical features from signal"""
    features = []
    for i in range(0, len(signal) - window_size, window_size):
        window = signal[i:i+window_size]
        features.append({
            'mean': np.mean(window),
            'std': np.std(window),
            'max': np.max(window),
            'min': np.min(window),
            'range': np.ptp(window),  # peak-to-peak
        })
    return pd.DataFrame(features)

features = extract_features(df['magnitude'].values)
print("\nExtracted Features:")
print(features.head())

# %%
# Visualize features
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(features['mean'])
plt.ylabel('Mean')
plt.title('Features Over Time')
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(features['std'])
plt.ylabel('Std Dev')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(features['range'])
plt.ylabel('Range')
plt.xlabel('Window Index')
plt.grid(True)

plt.tight_layout()
plt.show()

# %% [markdown]
"""
## PROJECT 2: Image Processing with NumPy

**Task:** Basic image manipulations using NumPy arrays
"""

# %%
# Create synthetic image (gradient + noise)
height, width = 256, 256

# Create meshgrid
x = np.linspace(0, 1, width)
y = np.linspace(0, 1, height)
X, Y = np.meshgrid(x, y)

# Create image
image = (X + Y) / 2 + np.random.randn(height, width) * 0.1
image = np.clip(image, 0, 1)  # Clip to [0, 1]

plt.figure(figsize=(8, 6))
plt.imshow(image, cmap='gray')
plt.colorbar()
plt.title('Original Image')
plt.show()

# %%
# Apply Gaussian blur (simple box filter)
from scipy.ndimage import gaussian_filter

blurred = gaussian_filter(image, sigma=3)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(blurred, cmap='gray')
plt.title('Blurred')
plt.colorbar()

plt.tight_layout()
plt.show()

# %%
# Edge detection (Sobel filter)
from scipy.ndimage import sobel

edges_x = sobel(blurred, axis=0)
edges_y = sobel(blurred, axis=1)
edges = np.hypot(edges_x, edges_y)

plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.imshow(blurred, cmap='gray')
plt.title('Blurred Image')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(edges_x, cmap='gray')
plt.title('Horizontal Edges')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray')
plt.title('Edge Magnitude')
plt.colorbar()

plt.tight_layout()
plt.show()

# %%
# Histogram equalization
hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 1])
cdf = hist.cumsum()
cdf_normalized = cdf / cdf[-1]

# Use CDF as lookup table
image_equalized = np.interp(image.flatten(), bins[:-1], cdf_normalized)
image_equalized = image_equalized.reshape(image.shape)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.hist(image.flatten(), bins=50, alpha=0.7)
plt.title('Original Histogram')
plt.xlabel('Pixel Value')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
plt.imshow(image_equalized, cmap='gray')
plt.title('Equalized')
plt.colorbar()

plt.tight_layout()
plt.show()

# %% [markdown]
"""
---
# CONGRATULATIONS! 🎉

You've completed the comprehensive NumPy course!

## What you've learned:
- ✅ NumPy fundamentals (arrays, indexing, operations)
- ✅ Advanced techniques (broadcasting, vectorization)
- ✅ Linear algebra operations
- ✅ Integration with Pandas, Matplotlib, SciPy
- ✅ Real-world applications

## Next steps:
1. Practice with Kaggle datasets
2. Build your own projects
3. Explore specialized libraries:
   - Scikit-learn for ML
   - TensorFlow/PyTorch for deep learning
   - OpenCV for computer vision

## Keep practicing! 🚀
The best way to master NumPy is to USE it daily.
"""

# %% [markdown]
"""
## Final Challenge: Robot Kinematics Dataset

Try this as your final project:
1. Download Robot Kinematics dataset from Kaggle
2. Load with Pandas
3. Analyze with NumPy
4. Visualize with Matplotlib
5. Apply SciPy for curve fitting

Good luck! 🤖
"""
