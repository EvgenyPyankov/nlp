import nltk

grammar = nltk.CFG.fromstring(
    """
    S -> NP VP
    PP -> P NP
    NP -> N | NP PP | A N | Det NP
    VP -> V NP | VP PP 
    N -> 'John' | 'man' | 'mountain'
    A -> 'high'
    V -> 'saw'
    Det -> 'the' | 'a'
    P -> 'on'
    """
)

str = "John saw the high man on the mountain"
sent = str.split(' ')
parser = nltk.ChartParser(grammar)
for tree in parser.parse(sent):
    print(tree)
