################################################################################
##                  AUXILIARY ROUTINES FOR GRAPH MANAGEMENT                   ##
################################################################################
# AUTHOR: Diego Alejandro Herrera Rojas
# DATE: 10/08/21
# DESCRIPTION: In this program, I use python-igraph for codifying a Heisenberg
# like Hamiltonian on an arbitrary graph. The point of these
# subroutines is to encode the exchange integrals and local
# external fields to a python object that represents an actual
# graph.

import igraph
from qiskit.circuit import Parameter


def generateGraph(spinInteraction, externalField):
    '''
    Function for generating
    graph from dictionary
    '''
    connections = spinInteraction.keys()
    exchangeIntegrals = list(spinInteraction.values())
    externalFields = list(externalField.values())
    g = igraph.Graph(connections)
    g.es['exchangeIntegrals'] = exchangeIntegrals
    g.es['paramExchangeIntegrals'] = [
        [
            Parameter('J_{}0'.format(edge.index)),
            Parameter('J_{}1'.format(edge.index)),
            Parameter('J_{}2'.format(edge.index))
        ] for edge in g.es
    ]
    g.vs['externalField'] = externalFields
    g.vs['paramExternalField'] = [
        Parameter('H_{}'.format(vertex.index))
        for vertex in g.vs
    ]
    return g


def getAdjacentEdges(g, edge):
    '''
    Function for Computing
    adjacent adjacent edges
    to given one
    '''
    (begin, end) = edge.tuple
    rawEdges = [
        *g.vs[begin].all_edges(),
        *g.vs[end].all_edges()
    ]
    return [
        adjacentEdge
        for adjacentEdge in rawEdges
        if adjacentEdge.index != edge.index
    ]


def colorGraph(g):
    '''
    Function for coloring
    a graph using a greedy
    algorithm
    '''
    g.es['color'] = [-1 for _ in range(len(g.es))]
    graphColors = [1]
    g.es[0]['color'] = graphColors[0]
    for edge in g.es:
        # get adjacent edges
        adjacentEdges = getAdjacentEdges(g, edge)
        # retrieve their colors
        adjacentColors = set(adjEdge['color'] for adjEdge in adjacentEdges)
        # compute color of current edge based on neighbors
        foundColor = edge['color'] not in adjacentColors and edge['color'] >= 0
        if not foundColor:
            for color in graphColors:
                if color not in adjacentColors:
                    edge['color'] = color
                    foundColor = True
                    break
        if not foundColor:
            graphColors.append(graphColors[-1]+1)
            edge['color'] = graphColors[-1]
    return graphColors


def colorMatching(g):
    '''
    Function for creating a
    matching of a graph based
    on coloring
    '''
    graphColors = colorGraph(g)
    matching = {}
    for color in graphColors:
        matching.update({
            color: g.es.select(lambda edge: edge['color'] == color)
        })
    return (matching, graphColors)


def graphRoutinesDemo():
    '''
    Function for illustrating
    how to use main module
    subroutines
    '''
    spinInteractions = {
        (0, 1): [0.5, 0.5, 0.5],
        (0, 3): [0.5, 0.5, 0.5],
        (0, 5): [0.5, 0.5, 0.5],
        (1, 2): [0.5, 0.5, 0.5],
        (1, 4): [0.5, 0.5, 0.5],
        (4, 3): [0.5, 0.5, 0.5],
        (2, 5): [0.5, 0.5, 0.5],
        (3, 2): [0.5, 0.5, 0.5],
        (4, 5): [0.5, 0.5, 0.5],
    }

    externalField = {
        0: [0, 0, 0],
        1: [0, 0, 0],
        2: [0, 0, 0],
        3: [0, 0, 0],
        4: [0, 0, 0],
        5: [0, 0, 0],
    }

    g = generateGraph(spinInteractions, externalField)

    matching, colors = colorMatching(g)

    for edge in g.es:
        print(edge.index, edge.tuple, edge['color'])

    print(colors)
    for color in colors:
        print(color, [edge.tuple for edge in matching[color]])
