import numpy as np

#base class for a Player
class Player:
    def __init__(self,  name,  id):
        self.name = name    #name of the player
        self.id = id                #the id also used to mark pieces on the board
     # perform a play action   
    def play(self,  board):
        print("player " + self.name + " is playing.")
        return 0
        
        
class HumanPlayer(Player):
    def play(self,  a_board):
        validMove = False;
        while not validMove:
            x, y = [int(x) for x in raw_input("Enter the coordinate to play (x, y): ").split(',')]
            #print x # x-coordinate to human user (index at 1)
            #print y #y-coordinate to human user (index at 1)
            if a_board.Get(y-1,x-1) == 0: 
                a_board.AddPiece(y-1, x-1,  self.id)
                validMove = True;
            else:
                print("Invalid play. Please choose again!")
                
        validMove = False;
        while not validMove:
            if a_board.Rotate() == 0:
                validMove = True;

class Board:
    def __init__(self):
        self.boardmtx = np.zeros((6,6))
        self.boardmtx[1,1]=1
        self.boardmtx[1,2]=2
        self.boardmtx[0,5]=1
        print(self.boardmtx)
        
    def AddPiece(self, x, y, value):
            self.boardmtx[x, y] = value
    def Get(self,  x,  y):
            return self.boardmtx[x, y]

    def Rotate(self):  #rotate quadrant counter-clockwise
      a = 0 # range start for y-coordinate
      b = 3 # range end for y-coordinate
      c = 3 # range start for x-coordinate
      d = 6 # range end for x-coordinate
      
      Quad = int(input("Which quadrant (1, 2, 3 or 4) to rotate? "))

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
      else:
          print ("Invalid quadrant.")
          return -1

      rotation = raw_input("Which direction to rotate (C or CC)?")
      print(rotation)
      #print rotation
      direction = -1
      if rotation == "C":
          direction = 3
      elif rotation == "CC":
          direction = 1
      else:
          print ("Invalid rotation.")
          return -1
                       
      Quadrant=self.boardmtx[a:b,c:d]
      #print Quadrant
      Quadrant = np.rot90(Quadrant,direction)
      #print Quadrant
      self.boardmtx[a:b,c:d]=Quadrant
      print(self.boardmtx)
      return 0


board = Board()
player1 = HumanPlayer("Pefik",  1)
player2 = HumanPlayer("Jarek", 2)

player1.play(board)
player2.play(board)
