#import heap methods and copy methods
from heapq import heappush,heappop, heapify
from copy import deepcopy

#Instances of the Node class saves information about the state
#Such as path cost(# of moves), parent, actions, evaluation function(fvalue) and the state itself or board configuration
#Saved the goal state's position to calculate the Manhattan distances heuristic function
class Node:
    def __init__(self,puzzle, goal, parent=None, move=""):
        self.state = puzzle
        self.goal = goal
        self.parent = parent
        self.pathCost = 0
        if parent is None:
            self.pathCost = 0
            self.actions = move
        else:
            self.pathCost = parent.pathCost+1
            self.actions = parent.actions + move + " "
        #Calculates the evaluation function f(n) as g(n)(path cost) + h(n)(heuristic-Manhattan distances)
        self.fvalue = self.pathCost + manhattanDist(self.state, self.goal)
    #Returns path cost of node
    def getPathCost(self):
        return self.pathCost
    #returns actions of node and its ancestors
    def getActions(self):
        return self.actions
    #Compare nodes through less than function to check if a node is less than another through its fvalue
    def __lt__(self, Node2):
        return self.fvalue <= Node2.fvalue
    #Compare nodes through equal to function to check if a node is equal to another through its board config
    def __eq__(self,Node2):
        return self.stringBoard() == Node2
    #Hashes the returning string value from stringBoard function
    def __hash__(self):
        return hash(self.stringBoard())
    #Make the board configuration into a single string to be hashed, strings are hashable
    def stringBoard(self):
        table = ""
        for row in self.state:
            for element in row:
                table += str(element)
        return table


#Conducts A* Search, keeps track of frontier priority queue(min-heap), explored set and number of nodes in search
class ASearch:
    def __init__(self, initialState, goalPos):
        self.frontier = []
        self.explored = set()
        heappush(self.frontier, Node(initialState,goalPos)) #pushes initial state into frontier
        self.nodes = 1
    #Algorithm for A* search where it returns 0 if frontier empty and solution not found or returns the state if it is the goal        
    def aSearch(self, goal):
        while True:
            if len(self.frontier) == 0:
                return 0
            #The node with the min fvalue is popped from the frontier with heappop
            chosen = heappop(self.frontier)
            if checkMatch(chosen.state,goal):
                return chosen
            #Adds chosen node to explored set if not added before
            self.explored.add(chosen.stringBoard())
            #Expands the chosen node by exploring possible moves and changed states as a result
            children = self.moves(chosen.state)
            for state in children:
                successor = Node(state[0],chosen.goal, chosen, state[1])
                #If successor node is not in explored set or frontier
                if successor.stringBoard() not in self.explored and successor not in self.frontier:
                    #Increment number of nodes
                    self.nodes += 1
                    #Add node to frontier heap while maintaining min-heap structure
                    heappush(self.frontier, successor)
                #Checks if successor is in the frontier
                elif successor in self.frontier:
                    for i in range(len(self.frontier)):
                        #Replace the node if the fvalue of the successor is less than the node in the frontier
                        if self.frontier[i] == successor and self.frontier[i].fvalue > successor.fvalue:
                            self.frontier[i] = successor
                            heapify(self.frontier)
    #Explores possible moves and adds changed states to an array then returns an array of the board configs
    def moves(self,state):
        #The possible moves for the blank position on the board at each location
        #If the blank position in at(0,0)(the upper left corner),the blank position can move to the right or down
        possibleMoves =[[['R','D'],['L','R','D'],['L','D']],
                        [['R','U','D'],['L','R','U','D'],['L','U','D']],
                        [['R','U'],['L','R','U'],['L','U']]]
        #Finds the location of the blank position
        blankpos1,blankpos2 = numBlankPos(state)
        #Holds the tuples of action and successor/child node given the state and location of the blank position
        expanded = []
        for move in possibleMoves[blankpos1][blankpos2]:
            #Creates new deep copy to the given state to find new state
            newState = deepcopy(state)
            #Right move
            if move == 'R':
                newState[blankpos1][blankpos2] = state[blankpos1][blankpos2+1]
                newState[blankpos1][blankpos2+1] = 0
                expanded.append((newState,'R'))
            #Left move
            if move == 'L':
                newState[blankpos1][blankpos2] = state[blankpos1][blankpos2-1]
                newState[blankpos1][blankpos2-1] = 0
                expanded.append((newState, 'L'))
            #Up move
            if move == 'U':
                newState[blankpos1][blankpos2] = state[blankpos1-1][blankpos2]
                newState[blankpos1-1][blankpos2] = 0
                expanded.append((newState, 'U'))
            #Down move
            if move == 'D':
                newState[blankpos1][blankpos2] = state[blankpos1+1][blankpos2]
                newState[blankpos1+1][blankpos2] = 0
                expanded.append((newState, 'D'))
        return expanded
    #Returns number of nodes in A* search
    def getNodes(self):
        return self.nodes
                        
        
    
#Calculates the manhattan distance given the state's board and the goal position dictionary
def manhattanDist(state,goal):
    manhattanDist = 0
    for row in range(len(state)):
        for col in range(len(state[row])):
            if(state[row][col] != 0):
                #Calculates the absolute values of the difference of the x-values and y-values and sums the difference 
                manhattanDist += abs(row-goal[state[row][col]][0])+abs(col-goal[state[row][col]][1])
    return manhattanDist
     
#Returns a tuple giving the position of the blank space in the state's board
def numBlankPos(board):
    pos1 = pos2 = 0
    for row in range(len(board)):
        for col in range(len(board[row])):
            if (board[row][col] == 0):
                pos1 = row
                pos2 = col
                break
    return pos1,pos2

#checks if the state board equals the goal board; if equals return true, else returns false
#Often checks if given state is goal state
def checkMatch(state,goal):
    for row in range(len(state)):
        for col in range(len(state[row])):
            if (state[row][col] != goal[row][col]):
                return False
    return True

#prints the board out in the specified manner
def printBoard(board):
  for row in range(len(board)):
        print('{} {} {}'.format(board[row][0],board[row][1], board[row][2]))


def main():
    #Open the input file and save into states variable
    states = open('Input7.txt', 'r')
    #Variable to keep track of the number of lines in the input file
    lines = 0
    #Arrays to hold the initial and goal states from the input file
    initialState = []
    goalState = []
    #Inputs values into the arrays
    for rows in states.readlines():
        rows = rows.strip() #strips whitespace
        if (lines < 3):
            initialState.append([int(i)for i in rows.split(' ')])
        elif (lines > 3):
            goalState.append([int(i)for i in rows.split(' ')])
        lines += 1
    states.close() #close input file
    #A dictionary that holds the position(coordinates) of each number in the goal state
    #The key is the number(1...8) and the value is a tuple of the coordinates
    posDict = {}
    #Creates dictionary by iterating through the goal state board
    for row in range(len(goalState)):
        for col in range(len(goalState[row])):
            if goalState[row][col] != 0:
                posDict[goalState[row][col]] = (row,col)
    #Create A* search instance, providing parameters of initial state and goal dictionary
    search = ASearch(initialState,posDict)
    #Conduct A* search, given goal board configuration
    result = search.aSearch(goalState)

    output = open("Output7.txt","w") #open file to write output to
    #If solution found and state returned
    if (result != 0):
        printBoard(initialState)
        print()
        printBoard(goalState)
        print()
        print(result.getPathCost())
        print(search.getNodes())
        print(result.getActions())
        #Writes output to output.txt file
        for row in range(len(initialState)):
            output.write('{} {} {}\n'.format(initialState[row][0],initialState[row][1], initialState[row][2]))
        output.write('\n')
        for row in range(len(initialState)):
            output.write('{} {} {}\n'.format(goalState[row][0],goalState[row][1], goalState[row][2]))
        output.write('\n')
        output.write(str(result.getPathCost()))
        output.write('\n')
        output.write(str(search.getNodes()))
        output.write('\n')
        output.write(str(result.getActions()))
    #Else if no solution found
    else:
        printBoard(initialState)
        print()
        printBoard(goalState)
        print()
        print("No solution found")
        #Writes output to output.txt file
        for row in range(len(initialState)):
            output.write('{} {} {}\n'.format(initialState[row][0],initialState[row][1], initialState[row][2]))
        output.write('\n')
        for row in range(len(initialState)):
            output.write('{} {} {}\n'.format(goalState[row][0],goalState[row][1], goalState[row][2]))
        output.write('\n')
        output.write('No solution found')
    output.close()    
#Runs the main function
if __name__ == '__main__':
    main()

