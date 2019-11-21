matrix=[[1,2,4,5],
        [6,7,8,9]] # матрица (2,4)

errs=[10,20]
    # матрица (2,1)

matr_res=[0]*len(matrix[0])

for elem in range(len(matrix[0])):
     #   matr_res[elem]=0
        for row in range(len(matrix)):
                matr_res[elem]+=errs[row] * matrix[row][elem]



print (matr_res)