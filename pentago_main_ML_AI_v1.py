import numpy as np
from random import *

#base class for a Player
class Player:
    def __init__(self,  name,  id):
        self.name = name    #name of the player
        self.id = id                #the id also used to mark pieces on the board
     # perform a play action   
    def play(self,  board):
        #print("player " + self.name + " is playing.")  #just in case
        return 0
        
        
class HumanPlayer(Player):
    def play(self,  a_board):
        #place a piece
        validMove = False
        while not validMove:
            x, y = [int(x) for x in raw_input("Enter the coordinate to play (x, y): ").split(',')]
            # x-coordinate to human user (index at 1)
            # y-coordinate to human user (index at 1)
            if a_board.Get(y-1,x-1) == 0: 
                a_board.AddPiece(y-1, x-1,  self.id)
                validMove = True
            else:
                print("Invalid play. Please choose again!")

        #rotate a quadrant
        Quad = 0
        while (Quad != 1) and (Quad != 2) and (Quad != 3) and (Quad != 4):
            Quad = int(input("Which quadrant (1, 2, 3 or 4) to rotate? "))
            #print ("Invalid quadrant.")
        
        direction = 0
        while direction != 3 and direction != 1:
            rotation = raw_input("Which direction to rotate (C or CC)?")
            if rotation == "C":
                  direction = 3
            elif rotation == "CC":
                  direction = 1
            else:
                  print("Invalid rotation.")
                  
        a_board.Rotate(Quad, direction)

class DummyAI(Player):
    def play(self,  a_board):
        #place a piece
        validMove = False
        while not validMove:
            x = randint(1,6)
            y = randint(1,6)
            if a_board.Get(y-1,x-1) == 0: 
                a_board.AddPiece(y-1, x-1,  self.id)
                validMove = True
            else:
                #print("Invalid play. Please choose again!")

        #rotate a quadrant
        Quad = randint(1,4)
        direction = randint(1,2)
        if direction == 2:
            direction = 3
                  
        a_board.Rotate(Quad, direction)
            
    

class Board:
    def __init__(self):
        self.boardmtx = np.zeros((6,6),dtype=np.int)
        #self.boardmtx[0,0] = 1  #for trouble-shooting game, start with a given board
        #self.boardmtx[0,1] = 1
        #self.boardmtx[0,2] = 1
        #self.boardmtx[0,3] = 1
        
        #print(self.boardmtx)

    def reset(self):
        self.boardmtx = np.zeros((6,6),dtpye=np.int)
    
    def AddPiece(self, x, y, value):    #change value at a position; needed for player methos play piece
            self.boardmtx[x, y] = value
    def Get(self,  x,  y):              #return value at a position; needed for player method play piece
            return self.boardmtx[x, y]

    def Rotate(self, Quad, direction):   #rotate quadrant
        a = 0 # range start for y-coordinate
        b = 3 # range end for y-coordinate
        c = 3 # range start for x-coordinate
        d = 6 # range end for x-coordinate
        if Quad == 1:  # top left  quadrant
              a = 0
              b = 3
              c = 0
              d = 3
        elif Quad == 2: #top right quadrant
              a = 0
              b = 3
              c = 3
              d = 6
        elif Quad == 3:  #bottom left quadrant
              a = 3
              b = 6
              c = 0
              d = 3
        elif Quad == 4:  #bottom right quadrant
              a = 3
              b = 6
              c = 3
              d = 6                      
        Quadrant=self.boardmtx[a:b,c:d]
        #print Quadrant
        Quadrant = np.rot90(Quadrant,direction)
        #print Quadrant
        self.boardmtx[a:b,c:d]=Quadrant
        #print(self.boardmtx)

    def GameEnd(self):
        diag1 = np.diag(self.boardmtx, k=-1).copy().tolist()+[0]
        diag2 = np.diag(self.boardmtx, k=1).copy().tolist()+[0]
        diag3 = np.diag(self.boardmtx).copy().tolist()
        diag4 = np.diag(np.rot90(self.boardmtx)).copy().tolist()   #off-diagonal, middle
        diag5 = np.diag(np.rot90(self.boardmtx), k=1).copy().tolist()+[0]
        diag6 = np.diag(np.rot90(self.boardmtx), k=-1).copy().tolist()+[0]
        diagonals = np.array([diag1, diag2, diag3, diag4, diag5, diag6])
        columns = np.rot90(self.boardmtx)
        IsGameOver = False
        def boardcheck(matrix):
            for row in matrix:
                if row[1] != 0:
                    if (row[1] == row[2]) and (row[1] == row[3]) and (row[1] == row[4]) and (row[1] == row[5] or row[0] == row[1]):
                        #print("Game over."), 
                        #if row[1] == 1:
                        #    print("Player 1 wins.")
                        #elif row[1] == 2:
                        #    print("Player 2 wins.")
                        return True
                        break
            return False
        
        IsGameOver = boardcheck(self.boardmtx)
        IsGameOver = IsGameOver or boardcheck(columns)
        IsGameOver = IsGameOver or boardcheck(diagonals)
        if IsGameOver is True:
            return 1
        else:
            return 0
        


board = Board()
player1 = HumanPlayer("Pefik",  1)
player2 = DummyAI("Rando", 2)

while True:
    player1.play(board)
    gameOver = board.GameEnd()
    if gameOver == 1:
        break

    player2.play(board)
    gameOver = board.GameEnd()
    if gameOver == 1:
        break
    
