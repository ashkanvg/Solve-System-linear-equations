import numpy as np

def isDefinitePositive(A): #with cholesky decomposition
    n = A.shape[0]  # size of matrix
    R = np.zeros((n, n))  # create empty n*n matrix
    if isSymmetric(A):
        for i in range(n):
            S = A[i, i] - (np.sum(R[:, i] ** 2))  # : means R[k,i] k < i
            if S < 0:  # if out matrix is not positive definite
                return False # maybe call gaus function
        return True
    else:
        return False

def lowerMatrix(L,b): #forward
    n = L.shape[0] # size of matrix
    y = np.zeros(n) # creat matrix n*1 and zero elements
    for i in range(n):
        y[i] = (b[i]- np.sum([L[i,j]*y[j] for j in range(i)]))/L[i,i] #formula for y_i in Ly=b
    return y

def upperMatrix(U,y):  #backward
    n = U.shape[0] # size of matrix
    x = np.zeros(n) # creat matrix n*1 and zero elements
    for i in range(n-1,-1,-1): # i = n-1 to 0
        x[i] = (y[i]- np.sum([U[i,j]*x[j] for j in range(i+1,n)]))/U[i,i] #formula for x_i in Ux=y
    return x

def cholesky(A):
    n = A.shape[0] #size of matrix
    R = np.zeros((n,n)) #create empty n*n matrix
    if isSymmetric(A):
        for i in range(n):
            S = A[i,i]-(np.sum(R[:,i]**2)) # : means R[k,i] k < i
            if S<0: #if out matrix is not positive definite
                print("matrix is not positive definite!")
                return False
            else: #if our matrix is positive definite
                R[i,i] = np.sqrt(S) #formula

            for j in range(i+1,n):
                R[i,j] = (A[i,j] - np.sum(R[:,i]*R[:,j]))/R[i,i]
        return R
    else:
        print("Matrix is not Symmetric, Then not POSITIVE DEFINITE")
        return False

def isSymmetric(A):
    if (A.T == A).all():
        print(1)
        return True
    else:
        return False

def gauss(A):
    B = A.copy()
    n = B.shape[0]
    intch = np.zeros(n, dtype=int) #for PLU
    for k in range(n-1): # n-1 because there is no need to use function for last row
        amax = np.max(np.absolute(B[k:n,k]))  #k:n --> columns k to n
        m = np.argmax(np.absolute(B[k:n,k])) + k # +k because index will start from k (k:n)
        # now : amax = A[m,k]  --> row: m and column: k
        if amax == 0:
            print("Matrix is Singular!")
            return False
        else:
            intch[k] = m
            if k!=m:
                B[[k,m]] = B[[m,k]] #change row m with row k
            B[k+1:n,k] = B[k+1:n,k]/B[k,k] #formula for get ZARIB ha
            #A[k+1:n,k] means every row of A in Columns k from row k+1 to n
            for i in range(k+1,n):
                B[i,k+1:n] = B[i,k+1:n] - B[i,k]*B[k,k+1:n]

    if B[n-1,n-1] == 0:
        print("Matrix is Singular!")
        return False
    else:
        intch[n-1] = n-1

    return B , intch

def LU(A):
    if gauss(A):
        B , intch = gauss(A)
        n = B.shape[0]
        U = np.triu(B)
        L = B - U + np.eye(n)

        P = np.eye(n)
        for k in range(n):
            m = intch[k]
            P[[k, m]] = P[[m, k]]

        return P,L,U
    else:
        return False

def solve(L,U,b):
    y = lowerMatrix(L, b)
    x = upperMatrix(U, y)
    return x

if __name__ == '__main__':
    print("________________ENTER_MATRIX_A________________")
    n = int(input("Enter the size of matrix 'A':"))
    if n<=0:
        print("size of matrix should be positive")
        exit(0)

    inputArray = []
    for i in range(n):  # A for loop for row entries
        a = []
        for j in range(n):  # A for loop for column entries
            print("Enter A[",i,"][",j,"]:")
            a.append(int(input()))
        inputArray.append(a)

    A = np.array(inputArray,dtype=float)
    print("________________YOUR_MATRIX________________")
    print(A)

    print("________________ENTER_MATRIX_b________________")
    inputB = []
    for i in range(n):  # A for loop for row entries
        #a = []
        print("Enter b[", i, "][0]:")
        inputB.append(int(input()))
        #inputB.append(a)
    b = np.array(inputB,dtype=float)
    print("________________YOUR_B________________")
    print(b)

    print("________________ANSWER________________")
    if isDefinitePositive(A):
        print("Matrix A is positive definite!")
        U = cholesky(A)
        L = U.T
        print("________________R________________")
        print(U)
        print("________________R_T________________")
        print(L)
        print("________________X________________")
        x = solve(L,U,b)
        print(x)
        # print(A)
        # print(L@U)
    else:
        if LU(A):
            print("Matrix A is not positive definite!")
            P,L,U = LU(A)
            b_hat = P@b
            A_hat = P@A
            print("note:PB=PAx (PA=LU) --> PB=LUx")
            print("________________A_HAT(PA)________________")
            print(A_hat)
            print("________________B_HAT(PB)________________")
            print(b_hat)
            print("________________U________________")
            print(U)
            print("________________L________________")
            print(L)
            print("________________P________________")
            print(P)
            print("________________X________________")
            x = solve(L,U,b_hat)
            print(x)

            #print(A_hat)
            #print(b_hat)
            #print(L @ U)

        exit(0)
    exit(0)