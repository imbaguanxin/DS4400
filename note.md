# DS4400

## Logistics

Office Hours: 4:30pm - 5:20, 310E WVH

TA: Thursdays 4-5 pm 162 WVH, Mondays 10:30 - 11:30 162 WVHs

Book: pattern recognition and machine learning, Christopher Bishop

Stanford CS229

### Grading

1. **4 homeworks**(40%)

 Analytical + Programming assign (Python) NO LATE HWS

2. **Project**(20%)

  **project team and proposal**: March 10th, 11:59 pm. 2 people

  **Final Project**: April 25th 11:59 pm

3. **midterms**(40%)

  **Midterm 1**: Feb 28

  **Midterm 2**: April 14

  1 sheet allowed

4. **class particition** (5%)

## 01.07

traditional programming: data和program，通过计算机得到output

ML: data, output -> program (model)

**Example:**

Data = { House1 = (sqft, category, location, ...), House2 = ...}
Output = {$xxx, $xxx ...}

**GOAL of ML:**

Function(house info) = price (even if we haven't seen.)

---

#### Machine learning types:

###### Supervised Machine Learning

desired response/output for each input is given.

1. **Classification:** from data to discrete status/outputs

  eg. spam fittering of email. (binary classification, only 2 possible outcome)

  eg. Image recognition. (multi class classification)

2.  **Regression:** From data to continuous outputs

  eg. House Price Cost/ stock value/ helthcare cost

  Data: historical info

  output: price/value

###### Unsupervised Machine Learning

no desired ouput, just find some interesting facts.

1. clustering (help manual classification)

2. Dimensionality reduction

###### Reinforcement Learning

* give a weak guide.

* between supervised and unsupervised

## 01.10

##### Linear Algebra contents:

1. inner product: <x,y>
2. outer product: x y^T (This is a matrix of n*m where x is a column vector of n elements and y is a column vector of m elements)
3. Identity matrix
4. Diagonal Matrix: D = diag(d1, d2, d3, ... , dn) =
| d1 | 0 |0 | ... | 0   |
| -- | -- | --|--|--|
| 0 | d2 | 0 | ... | 0 |
| 0 | 0 | d3 | ... | 0 |
| ... | ... | ... | ... | ... |
| 0 | 0 | 0 | ... | dn |
5. symmetric: A^T = A
6. antisymmetric: A^T = -A
7. trace(A) = sum of elements on the diagonal. trace(A^T) = trace(A); trace(kA) = ktrace(A); trace(AB) = trace(BA)
