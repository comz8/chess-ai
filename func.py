def king_moves(x, y):
    moves = [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx, dy) != (0, 0)]
    return [(x_, y_) for x_, y_ in moves if 0 <= x_ < 8 and 0 <= y_ < 8]

def queen_moves(x, y):
    return rook_moves(x, y) + bishop_moves(x, y)

def rook_moves(x, y):
    moves = [(x, i) for i in range(8) if i != y] + [(i, y) for i in range(8) if i != x]
    return moves

def bishop_moves(x, y):
    moves = [(x + i, y + i) for i in range(1, 8)] + [(x - i, y - i) for i in range(1, 8)] + \
            [(x + i, y - i) for i in range(1, 8)] + [(x - i, y + i) for i in range(1, 8)]
    return [(x_, y_) for x_, y_ in moves if 0 <= x_ < 8 and 0 <= y_ < 8]

def knight_moves(x, y):
    moves = [(x + 2, y + 1), (x + 2, y - 1), (x - 2, y + 1), (x - 2, y - 1),
             (x + 1, y + 2), (x + 1, y - 2), (x - 1, y + 2), (x - 1, y - 2)]
    return [(x_, y_) for x_, y_ in moves if 0 <= x_ < 8 and 0 <= y_ < 8]

def pawn_moves(x, y, color):
    direction = 1 if color == 'white' else -1
    moves = [(x, y + direction)]
    if (y == 1 and color == 'white') or (y == 6 and color == 'black'):
        moves.append((x, y + 2 * direction))
    return [(x_, y_) for x_, y_ in moves if 0 <= x_ < 8 and 0 <= y_ < 8]


# 예시 실행
print("King moves:", king_moves(4, 4))
print("Queen moves:", queen_moves(4, 4))
print("Rook moves:", rook_moves(4, 4))
print("Bishop moves:", bishop_moves(4, 4))
print("Knight moves:", knight_moves(4, 4))
print("Pawn moves:", pawn_moves(4, 1, 'white'))
