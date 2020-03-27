import numpy as np

# unary_2x2 = [
#     {"name": "A : B  ::  C : ?", "value": [(0, 0), (0, 1), (1, 0), (1, 1)], "type": "unary_2x2", "chld_name": None, "shape": (2, 2)},
#     {"name": "A : C  ::  B : ?", "value": [(0, 0), (1, 0), (0, 1), (1, 1)], "type": "unary_2x2", "chld_name": None, "shape": (2, 2)}
# ]
#
# unary_3x3 = [
#     # 1. unary_3x3_row_H
#     {"name": "A : B  ::  D : E  :::  E : F  ::  H : ?", "value": [[(0, 0), (0, 1), (1, 0), (1, 1)], [(1, 1), (1, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},  # row analogies
#     {"name": "B : C  ::  E : F  :::  E : F  ::  H : ?", "value": [[(0, 1), (0, 2), (1, 1), (1, 2)], [(1, 1), (1, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "D : E  ::  G : H  :::  E : F  ::  H : ?", "value": [[(1, 0), (1, 1), (2, 0), (2, 1)], [(1, 1), (1, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#     # {"name": "unary_3x3_row_H", "value": [[(0, 0), (0, 1), (1, 0), (1, 1)],
#     #                                       [(0, 1), (0, 2), (1, 1), (1, 2)],
#     #                                       [(1, 0), (1, 1), (2, 0), (2, 1)],
#     #                                       [(1, 1), (1, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#     # 2. unary_3x3_row_G
#     {"name": "A : C  ::  D : F  :::  D : F  ::  G : ?", "value": [[(0, 0), (0, 2), (1, 0), (1, 2)], [(1, 0), (1, 2), (2, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "B : A  ::  E : D  :::  D : F  ::  G : ?", "value": [[(0, 1), (0, 0), (1, 1), (1, 0)], [(1, 0), (1, 2), (2, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "E : D  ::  H : G  :::  D : F  ::  G : ?", "value": [[(1, 1), (1, 0), (1, 0), (1, 2)], [(1, 0), (1, 2), (2, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#     # {"name": "unary_3x3_row_G", "value": [[(0, 0), (0, 2), (1, 0), (1, 2)],
#     #                                       [(0, 1), (0, 0), (1, 1), (1, 0)],
#     #                                       [(1, 1), (1, 0), (1, 0), (1, 2)],
#     #                                       [(1, 0), (1, 2), (2, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#     # 3. unary_3x3_col_F
#     {"name": "A : D  ::  B : E  :::  E : H  ::  F : ?", "value": [[(0, 0), (0, 1), (1, 0), (1, 1)], [(1, 1), (1, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : C  ::  B : ?"},  # col analogies
#     {"name": "D : G  ::  E : H  :::  E : H  ::  F : ?", "value": [[(1, 0), (1, 1), (2, 0), (2, 1)], [(1, 1), (1, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : C  ::  B : ?"},
#     {"name": "B : E  ::  C : F  :::  E : H  ::  F : ?", "value": [[(0, 1), (0, 2), (1, 1), (1, 2)], [(1, 1), (1, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : C  ::  B : ?"},
#
#     # {"name": "unary_3x3_col_F", "value": [[(0, 0), (0, 1), (1, 0), (1, 1)],
#     #                                       [(1, 0), (1, 1), (2, 0), (2, 1)],
#     #                                       [(0, 1), (0, 2), (1, 1), (1, 2)],
#     #                                       [(1, 1), (1, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : C  ::  B : ?"},
#
#     # 4. unary_3x3_col_C
#     {"name": "A : G  ::  B : H  :::  B : H  ::  C : ?", "value": [[(0, 0), (2, 0), (0, 1), (2, 1)], [(0, 1), (0, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "D : A  ::  E : B  :::  B : H  ::  C : ?", "value": [[(1, 0), (0, 0), (1, 1), (0, 1)], [(0, 1), (0, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "E : B  ::  F : C  :::  B : H  ::  C : ?", "value": [[(1, 1), (0, 1), (1, 2), (0, 2)], [(0, 1), (0, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#     # {"name": "unary_3x3_col_C", "value": [[(0, 0), (2, 0), (0, 1), (2, 1)],
#     #                                       [(1, 0), (0, 0), (1, 1), (0, 1)],
#     #                                       [(1, 1), (0, 1), (1, 2), (0, 2)],
#     #                                       [(0, 1), (0, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#
#     # 5. unary_3x3_row_G_rpl, replacing A to the right of its row and ? to the left of its row
#     {"name": "C : B  ::  E : D  :::  E : D  ::  G : ?", "value": [[(0, 2), (0, 1), (1, 1), (1, 0)], [(1, 1), (1, 0), (2, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "A : C  ::  F : E  :::  E : D  ::  G : ?", "value": [[(0, 0), (0, 2), (1, 2), (1, 1)], [(1, 1), (1, 0), (2, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "F : E  ::  H : G  :::  E : D  ::  G : ?", "value": [[(1, 2), (1, 1), (2, 1), (2, 0)], [(1, 1), (1, 0), (2, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#     # {"name": "unary_3x3_row_G_rpl", "value": [[(0, 2), (0, 1), (1, 1), (1, 0)],
#     #                                           [(0, 0), (0, 2), (1, 2), (1, 1)],
#     #                                           [(1, 2), (1, 1), (2, 1), (2, 0)],
#     #                                           [(1, 1), (1, 0), (2, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#     # 6. unary_3x3_row_H_rpl, replacing A to the right of its row and ? to the left of its row
#     {"name": "A : B  ::  F : D  :::  F : D  ::  H : ?", "value": [[(0, 0), (0, 1), (1, 2), (1, 0)], [(1, 2), (1, 0), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "C : A  ::  E : F  :::  F : D  ::  H : ?", "value": [[(0, 2), (0, 0), (1, 1), (1, 2)], [(1, 2), (1, 0), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "E : F  ::  G : H  :::  F : D  ::  H : ?", "value": [[(1, 1), (1, 2), (2, 0), (2, 1)], [(1, 2), (1, 0), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#     # {"name": "unary_3x3_row_H_rpl", "value": [[(0, 0), (0, 1), (1, 2), (1, 0)],
#     #                                           [(0, 2), (0, 0), (1, 1), (1, 2)],
#     #                                           [(1, 1), (1, 2), (2, 0), (2, 1)],
#     #                                           [(1, 2), (1, 0), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#
#      # unary_3x3_col_D_rpl, replacing A to the bottom of it col and ? to the top of its col
#     {"name": "F : H  ::  E : G  :::  E : G  ::  D : ?", "value": [[(1, 2), (2, 1), (1, 1), (2, 0)], [(1, 1), (2, 0), (1, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "A : F  ::  C : E  :::  E : G  ::  D : ?", "value": [[(0, 0), (1, 2), (0, 2), (1, 1)], [(1, 1), (2, 0), (1, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "C : E  ::  B : D  :::  E : G  ::  D : ?", "value": [[(0, 2), (1, 1), (0, 1), (1, 0)], [(1, 1), (2, 0), (1, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#     # {"name": "unary_3x3_col_rpl_row_D", "value": [[(1, 2), (2, 1), (1, 1), (2, 0)],
#     #                                               [(0, 0), (1, 2), (0, 2), (1, 1)],
#     #                                               [(0, 2), (1, 1), (0, 1), (1, 0)],
#     #                                               [(1, 1), (2, 0), (1, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#     {"name": "A : H  ::  C : G  :::  C : G  ::  B : ?", "value": [[(0, 0), (2, 1), (0, 2), (2, 0)], [(0, 2), (2, 0), (0, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#     {"name": "A : F  ::  G : C  :::  G : C  ::  D : ?", "value": [[(0, 0), (1, 2), (2, 0), (0, 2)], [(2, 0), (0, 2), (1, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"}, # row analogies after replacing the first and last entry in col direction
#     {"name": "A : H  ::  G : E  :::  E : C  ::  B : ?", "value": [[(0, 0), (2, 1), (2, 0), (1, 1)], [(1, 1), (0, 2), (0, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "H : F  ::  E : C  :::  E : C  ::  B : ?", "value": [[(2, 1), (1, 2), (1, 1), (0, 2)], [(1, 1), (0, 2), (0, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "G : E  ::  D : B  :::  E : C  ::  B : ?", "value": [[(2, 0), (1, 1), (1, 0), (0, 1)], [(1, 1), (0, 2), (0, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#     # {"name": "unary_3x3_row_rpl_col_B", "value": [[(0, 0), (2, 1), (2, 0), (1, 1)],
#     #                                               [(2, 0), (1, 1), (1, 0), (0, 1)],
#     #                                               [(2, 1), (1, 2), (1, 1), (0, 2)],
#     #                                               [(1, 1), (0, 2), (0, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#     {"name": "A : D  ::  H : B  :::  H : B  ::  F : ?", "value": [[(0, 0), (1, 0), (2, 1), (0, 1)], [(2, 1), (0, 1), (1, 2), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"}, # col analogies after replacing the first and last entry in col direction
#     {"name": "A : G  ::  H : E  :::  E : B  ::  C : ?", "value": [[(0, 0), (2, 0), (2, 1), (1, 1)], [(1, 1), (0, 1), (0, 2), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "H : E  ::  F : C  :::  E : B  ::  C : ?", "value": [[(2, 1), (1, 1), (1, 2), (0, 2)], [(1, 1), (0, 1), (0, 2), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "G : D  ::  E : B  :::  E : B  ::  C : ?", "value": [[(2, 0), (1, 0), (1, 1), (0, 1)], [(1, 1), (0, 1), (0, 2), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#     # {"name": "unary_3x3_col_rpl_col_C", "value": [[(0, 0), (2, 0), (2, 1), (1, 1)],
#     #                                               [(2, 1), (1, 1), (1, 2), (0, 2)],
#     #                                               [(2, 0), (1, 0), (1, 1), (0, 1)],
#     #                                               [(1, 1), (0, 1), (0, 2), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#     {"name": "F : A  ::  G : E  :::  G : E  ::  B : ?", "value": [[(1, 0), (0, 0), (2, 0), (1, 1)], [(2, 0), (1, 1), (0, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},  # dgn analogies
#     {"name": "H : A  ::  C : E  :::  C : E  ::  D : ?", "value": [[(2, 1), (0, 0), (0, 2), (1, 1)], [(0, 2), (1, 1), (1, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "C : D  ::  G : B  :::  G : B  ::  E : ?", "value": [[(0, 2), (1, 0), (2, 0), (0, 1)], [(2, 0), (0, 1), (1, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#     {"name": "H : D  ::  F : B  :::  F : B  ::  A : ?", "value": [[(2, 1), (1, 0), (1, 2), (0, 1)], [(1, 2), (0, 1), (0, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
#
#
# ]
#
# binary_2x3 = [
#     {"name": "A : B : C  ::  D : E : ?", "value": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], "type": "binary_2x3", "chld_name": None, "shape": (2, 3)}
# ]
#
# binary_3x3 = [
#     {"name": "A : B : C  ::  D : E : F  :::  D : E : F  ::  G : H : ?", "value": [[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]], "type": "binary_3x3", "chld_name": "A : B : C  ::  D : E : ?"},
#     {"name": "A : D : G  ::  B : E : H  :::  B : E : H  ::  C : F : ?", "value": [[(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)], [(0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]], "type": "binary_3x3", "chld_name": "A : D : D  ::  B : E : ?"},
# ]
#
#


def generate_analogies(symbol_matrix, symbol_to_coord, tuple_n, link_symbol):

    row_n, col_n = np.array(symbol_matrix).shape

    row_analogies = None
    if col_n >= tuple_n:
        row_analogies = generate_row_analogies(symbol_matrix, symbol_to_coord, tuple_n, link_symbol)

    col_analogies = None
    if row_n >= tuple_n:
        col_analogies = generate_col_analogies(symbol_matrix, symbol_to_coord, tuple_n, link_symbol)

    return row_analogies, col_analogies


def generate_col_analogies(symbol_matrix, symbol_to_coord, tuple_n, link_symbol):

    tuple_symbol_matrix, tuple_symbol_to_coord = generate_row_analogies(np.array(symbol_matrix).transpose().tolist(),
                                  symbol_to_coord, tuple_n, link_symbol)
    tuple_symbol_matrix = np.array(tuple_symbol_matrix).transpose().tolist()
    return tuple_symbol_matrix, tuple_symbol_to_coord


def generate_row_analogies(symbol_matrix, symbol_to_coord, tuple_n, link_symbol):

    row_n, col_n = np.array(symbol_matrix).shape

    tuple_symbol_matrix = []
    tuple_coord_matrix = []
    tuple_symbol_to_coord = {}
    for ii in range(row_n):

        tuple_symbol_matrix.append([])
        tuple_coord_matrix.append([])
        for jj in range(col_n - tuple_n + 1):

            symbols = []
            tuple_coords = []
            for kk in range(tuple_n):
                symbol = symbol_matrix[ii][jj + kk]
                symbols.append(symbol)
                tuple_coords.extend(symbol_to_coord.get(symbol))

            tuple_symbol = link_symbol.join(symbols)
            tuple_symbol_matrix[ii].append(tuple_symbol)
            tuple_coord_matrix[ii].append(tuple_coords)
            tuple_symbol_to_coord[tuple_symbol] = tuple_coords

    return tuple_symbol_matrix, tuple_symbol_to_coord


def collapse(args, link_symbol, type, chld_name, shape):

    anlgs = []
    allinone_anlgs = []
    for arg in args:

        symbol_matrix, symbol_to_coord = arg
        symbols = np.array(symbol_matrix).flatten().tolist()

        tail = None
        for s in symbols:
            if '?' in s:
                tail = s
                break
        symbols.remove(tail)

        for head in symbols:
            name = head + link_symbol + tail
            value = symbol_to_coord.get(head) + symbol_to_coord.get(tail)
            anlgs.append({
                "name": name,
                "value": value,
                "type": type,
                "chld_name": chld_name,
                "chld_n": 2,
                "shape": shape
            })

        if len(symbols) > 1:
            all_name = link_symbol.join(symbols) + link_symbol + tail
            all_value = []
            for s in symbols + [tail]:
                all_value.extend(symbol_to_coord.get(s))
            allinone_anlgs.append({
                "name": all_name,
                "value": all_value,
                "type": type,
                "chld_name": chld_name,
                "chld_n": len(symbols) + 1,
                "shape": shape
            })

    return anlgs + allinone_anlgs


def get_matrix_analogies(symbol_matrix, symbol_to_coord, tuple_n, link_symbol, order_n):

    if not isinstance(tuple_n, list):
        tuple_n = [tuple_n] * (order_n - 1)

    order_link = link_symbol
    args = [(symbol_matrix, symbol_to_coord)]
    for order in range(order_n - 1):

        new_args = []
        for arg in args:

            row, col = generate_analogies(*arg, tuple_n[order], order_link)
            if row is not None:
                new_args.append(row)
            if col is not None:
                new_args.append(col)

        order_link += link_symbol
        args = new_args

    row_n, col_n = np.array(symbol_matrix).shape
    if 2 == tuple_n[0]:
        anlg_type = "unary_" + str(row_n) + 'x' + str(col_n)
    elif 3 == tuple_n[0]:
        anlg_type = "binary_" + str(row_n) + 'x' + str(col_n)
    else:
        anlg_type = "X_" + str(row_n) + 'x' + str(col_n)

    if "unary_3x3" == anlg_type:
        chld_name = "A:B::C:?"
    elif "binary_3x3" == anlg_type:
        chld_name = "A:B:C::D:E:?"
    else:
        chld_name = None

    return collapse(args, order_link, anlg_type, chld_name, (row_n, col_n))


def remove_redundant_ones(anlgs):

    thin_anlgs = []
    for anlg in anlgs:
        name = anlg.get("name")

        found = False
        for t_anlg in thin_anlgs:
            t_name = t_anlg.get("name")
            if name == t_name:
                found = True
                break

        if not found:
            thin_anlgs.append(anlg)

    return thin_anlgs

    # non_double_line_angls = []
    # for anlg in thin_anlgs:
    #     value = anlg.get("value")
    #     double_line = False
    #     kk = 0
    #     for ii in range(len(value) - 1):
    #         if value[ii] == value[ii + 1]:
    #             kk += 1
    #         if kk == 2:
    #             double_line = True
    #             break
    #     if not double_line:
    #         non_double_line_angls.append(anlg)
    #
    # return non_double_line_angls


symbol_to_coord_2x2 = {
    'A': [(0, 0)],
    'B': [(0, 1)],
    'C': [(1, 0)],
    '?': [(1, 1)]
}

symbol_to_coord_3x3 = {
    'A': [(0, 0)],
    'B': [(0, 1)],
    'C': [(0, 2)],
    'D': [(1, 0)],
    'E': [(1, 1)],
    'F': [(1, 2)],
    'G': [(2, 0)],
    'H': [(2, 1)],
    '?': [(2, 2)]
}


matrices_2x2 = [[['A', 'B'],
                 ['C', '?']],

                [['B', 'A'],
                 ['C', '?']],

                [['C', 'B'],
                 ['A', '?']]]
unary_2x2 = []
for m in matrices_2x2:
    anlgs = get_matrix_analogies(m, symbol_to_coord_2x2, 2, ':', 2)
    anlgs = remove_redundant_ones(anlgs)
    unary_2x2.extend(anlgs)

matrices_3x3 = [  # redundancy exists,
                  # since replace rows won't affect row analogies and replace cols won't affect col analogies
                  # you may want to filter out these redundant analogies later.
                [['A', 'B', 'C'],  # original
                 ['D', 'E', 'F'],
                 ['G', 'H', '?']],

                [['B', 'A', 'C'],  # original, replace cols
                 ['E', 'D', 'F'],
                 ['H', 'G', '?']],

                [['D', 'E', 'F'],  # original, replace rows
                 ['A', 'B', 'C'],
                 ['G', 'H', '?']],

                [['E', 'D', 'F'],  # original, replace rows and cols
                 ['B', 'A', 'C'],
                 ['H', 'G', '?']],

                [['A', 'C', 'B'],  # replace the first and the last entries in row
                 ['F', 'E', 'D'],
                 ['H', 'G', '?']],

                [['F', 'E', 'D'],  # replace the first and the last entries in row, and replace rows
                 ['A', 'C', 'B'],
                 ['H', 'G', '?']],

                [['C', 'A', 'B'],  # replace the first and the last entries in row, and replace cols
                 ['E', 'F', 'D'],
                 ['G', 'H', '?']],

                [['E', 'F', 'D'],  # replace the first and the last entries in row, and replace cols and rows
                 ['C', 'A', 'B'],
                 ['G', 'H', '?']],

                [['A', 'H', 'F'],  # replace the first and the last entries in col
                 ['G', 'E', 'C'],
                 ['D', 'B', '?']],

                [['G', 'E', 'C'],  # replace the first and the last entries in col, and replace rows
                 ['A', 'H', 'F'],
                 ['D', 'B', '?']],

                [['H', 'A', 'F'],  # replace the first and the last entries in col, and replace cols
                 ['E', 'G', 'C'],
                 ['B', 'D', '?']],

                [['E', 'G', 'C'],  # replace the first and the last entries in col, and replace cols and rows
                 ['H', 'A', 'F'],
                 ['B', 'D', '?']],

                [['H', 'F', 'A'],  # I don't how to change the original to this one for now. But it definitely explain two diagonal directions very well.
                 ['C', 'G', 'E'],  # Let's call it strange one.
                 ['D', 'B', '?']],

                [['C', 'G', 'E'],  # strange one, replace rows.
                 ['H', 'F', 'A'],
                 ['D', 'B', '?']],

                [['F', 'H', 'A'],  # strange one, replace cols.
                 ['G', 'C', 'E'],
                 ['B', 'D', '?']],

                [['G', 'C', 'E'],  # strange one, replace rows and cols.
                 ['F', 'H', 'A'],
                 ['B', 'D', '?']],
]

unary_3x3 = []
binary_3x3 = []
for m in matrices_3x3:
    unary_anlgs = get_matrix_analogies(m, symbol_to_coord_3x3, 2, ':', 3)
    unary_anlgs = remove_redundant_ones(unary_anlgs)
    unary_3x3.extend(unary_anlgs)
    binary_anlgs = get_matrix_analogies(m, symbol_to_coord_3x3, [3, 2], ':', 3)
    binary_anlgs = remove_redundant_ones(binary_anlgs)
    binary_3x3.extend(binary_anlgs)


symbol_to_coord_2x3 = {
    "A": (0, 0),
    "B": (0, 1),
    "C": (0, 2),
    "D": (1, 0),
    "E": (1, 1),
    "?": (1, 2)
}


matrix_2x3 = [[['A', 'B', 'C'],
               ['D', 'E', '?']]]

binary_2x3 = []
for m in matrix_2x3:
    binary_anlgs = get_matrix_analogies(m, symbol_to_coord_3x3, 3, ':', 2)
    binary_anlgs = remove_redundant_ones(binary_anlgs)
    binary_2x3.extend(binary_anlgs)


all_anlgs = unary_2x2 + unary_3x3 + binary_2x3 + binary_3x3


def get_anlg(anlg_name):
    for anlg in all_anlgs:
        if anlg_name == anlg.get("name"):
            return anlg


