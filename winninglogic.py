import numpy as np
boogie = np.matrix( [[1,0,0,0,1,0],[2,2,0,1,0,0],[1,0,2,0,0,0],[1,1,1,2,1,1],[1,0,0,0,2,0],[1,0,1,0,1,2]])
print boogie


def GameEnd(board):
    diag1 = np.diag(board, k=-1).copy().tolist()+[0]
    diag2 = np.diag(board, k=1).copy().tolist()+[0]
    diag3 = np.diag(board).copy().tolist()
    diag4 = np.diag(np.rot90(board)).copy().tolist()   #off-diagonal, middle
    diag5 = np.diag(np.rot90(board), k=1).copy().tolist()+[0]
    diag6 = np.diag(np.rot90(board), k=-1).copy().tolist()+[0]
    #print diag1, diag2, diag3, diag4, diag5, diag6
    diagonals = np.matrix([diag1, diag2, diag3, diag4, diag5, diag6])
    columns = np.rot90(board)
    #bigone = np.matrix([diagonals],[board],[columns])
    #print bigone
    #print check
    def boardcheck(matrix):
        for row in matrix:
            if row [0,1] != 0:
                if row[0,1] == row[0,2] and row[0,1] == row[0,3] and row[0,1] == row[0,4] and (row[0,1] == row[0,5] or row[0,0] == row [0,1]):
                    print("Game over."), 
                    if row[0,1] == 1:
                        print("Player 1 wins.")
                    elif row[0,1] == 2:
                        print("Player 2 wins.")
                    break
    boardcheck(board)
    boardcheck(columns)
    boardcheck(diagonals)

GameEnd(boogie)
