
unary_2x2 = [
    {"name": "A : B  ::  C : ?", "value": [(0, 0), (0, 1), (1, 0), (1, 1)], "type": "unary_2x2", "chld_name": None, "shape": (2, 2)},
    {"name": "A : C  ::  B : ?", "value": [(0, 0), (1, 0), (0, 1), (1, 1)], "type": "unary_2x2", "chld_name": None, "shape": (2, 2)}
]

unary_3x3 = [
    {"name": "A : B  ::  D : E  :::  E : F  ::  H : ?", "value": [[(0, 0), (0, 1), (1, 0), (1, 1)], [(1, 1), (1, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},  # row analogies
    {"name": "B : C  ::  E : F  :::  E : F  ::  H : ?", "value": [[(0, 1), (0, 2), (1, 1), (1, 2)], [(1, 1), (1, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
    {"name": "D : E  ::  G : H  :::  E : F  ::  H : ?", "value": [[(1, 0), (1, 1), (2, 0), (2, 1)], [(1, 1), (1, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
    {"name": "A : C  ::  D : F  :::  D : F  ::  G : ?", "value": [[(0, 0), (0, 2), (1, 0), (1, 2)], [(1, 0), (1, 2), (2, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},

    {"name": "A : D  ::  B : E  :::  E : H  ::  F : ?", "value": [[(0, 0), (0, 1), (1, 0), (1, 1)], [(1, 1), (1, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : C  ::  B : ?"},  # col analogies
    {"name": "D : G  ::  E : H  :::  E : H  ::  F : ?", "value": [[(1, 0), (1, 1), (2, 0), (2, 1)], [(1, 1), (1, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : C  ::  B : ?"},
    {"name": "B : E  ::  C : F  :::  E : H  ::  F : ?", "value": [[(0, 1), (0, 2), (1, 1), (1, 2)], [(1, 1), (1, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : C  ::  B : ?"},
    {"name": "A : G  ::  B : H  :::  B : H  ::  C : ?", "value": [[(0, 0), (0, 1), (2, 0), (2, 1)], [(0, 1), (0, 2), (2, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : C  ::  B : ?"},

    {"name": "F : A  ::  G : E  :::  G : E  ::  B : ?", "value": [[(1, 0), (0, 0), (2, 0), (1, 1)], [(2, 0), (1, 1), (0, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},  # dgn analogies
    {"name": "H : A  ::  C : E  :::  C : E  ::  D : ?", "value": [[(2, 1), (0, 0), (0, 2), (1, 1)], [(0, 2), (1, 1), (1, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
    {"name": "C : D  ::  G : B  :::  G : B  ::  E : ?", "value": [[(0, 2), (1, 0), (2, 0), (0, 1)], [(2, 0), (0, 1), (1, 1), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"},
    {"name": "H : D  ::  F : B  :::  F : B  ::  A : ?", "value": [[(2, 1), (1, 0), (1, 2), (0, 1)], [(1, 2), (0, 1), (0, 0), (2, 2)]], "type": "unary_3x3", "chld_name": "A : B  ::  C : ?"}
]

binary_3x2 = [
    {"name": "A : D : G  ::  B : E : ?", "value": [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)], "type": "binary_3x2", "chld_name": None, "shape": (3, 2)}
]

binary_2x3 = [
    {"name": "A : B : C  ::  D : E : ?", "value": [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], "type": "binary_2x3", "chld_name": None, "shape": (2, 3)}
]

binary_3x3 = [
    {"name": "A : B : C  ::  D : E : F  :::  D : E : F  ::  G : H : ?", "value": [[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]], "type": "binary_3x3", "chld_name": "A : B : C  ::  D : E : ?"},
    {"name": "A : D : G  ::  B : E : H  :::  B : E : H  ::  C : F : ?", "value": [[(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)], [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2)]], "type": "binary_3x3", "chld_name": "A : D : G  ::  B : E : ?"}
]


all_anlgs = unary_2x2 + unary_3x3 + binary_3x2 + binary_2x3 + binary_3x3


def get_anlg(anlg_name):
    for anlg in all_anlgs:
        if anlg_name == anlg.get("name"):
            return anlg


