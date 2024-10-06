"""Main file, intended to be run"""

#units are m, kg, s

mass = 0.2241   # kg

LO_HP = 0.178   # m
G_YN = 0.217    # m

A_r = 0.00378   # m^2
A_w = 0.00364   # m^2

def static(material : str):
    """Does all the static friction stuff"""
    
    raise NotImplementedError

def kinetic(material : str):
    """Does all the kinetic friction stuff"""
    
    raise NotImplementedError

def main():
    """Everything all together"""

    raise NotImplementedError


if __name__ == "__main__":
    main()