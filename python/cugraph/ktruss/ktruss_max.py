# Copyright (c) 2019, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cugraph.ktruss import ktruss_max_wrapper


def ktruss_max(G):
    """
    Finds the maximal k-truss of a graph.

    The k-truss of a graph is subgraph where each edge is part  of  at least
    (k−2) triangles. The maximal k-truss in a graph, denoted by
    k=k_max is the largest k-truss in the graph where the set of satisfying
    edges is not empty. k-trusses are used for finding tighlty knit groups of
    vertices in a graph.
    A k-truss is a relaxation of a k-clique in the graph and was define in
    [1]. Finding cliques is computationally demanding and finding the maximal
    k-clique is known to be NP-Hard.

    In contrast, finding a k-truss is computationally tractable as its
    key building block, namely triangle counting counting, can be
    executed in polnymomial time. Typically, it takes many iterations of
    triangle counting to find the k-truss of a graph.
    Yet these iterations operate on a weakly monotonically shrinking graph.
    Therefore, finding the k-truss of a graph can be done in a fairly
    reasonable amount of time.
    The solution in cuGraph is based on a GPU algorithm first shown
    in [2] and uses the triangle counting algoritm from [3].

    [1] Cohen, J.,
    "Trusses: Cohesive subgraphs for social network analysis"
    National security agency technical report, 2008

    [2] O. Green, J. Fox, E. Kim, F. Busato, et al.
    “Quickly Finding a Truss in a Haystack”
    IEEE High Performance Extreme Computing Conference (HPEC), 2017
    https://doi.org/10.1109/HPEC.2017.8091038

    [3] O. Green, P. Yalamanchili, L.M. Munguia,
    “Fast Triangle Counting on GPU”
    Irregular Applications: Architectures and Algorithms (IA3), 2014


    Parameters
    ----------
    G : cuGraph.Graph
        cuGraph graph descriptor with connectivity information. k-Trusses are
        defined for only undirected graphs as they are defined for
        undirected triangle in a graph.

    Returns
    -------
    k_max : int
        The largest k in the graph s.t. a non-empty k-truss in the
        graph exists.


    Examples
    --------
    >>> M = cudf.read_csv('datasets/karate.csv', delimiter=' ',
    >>>                   dtype=['int32', 'int32', 'float32'], header=None)
    >>> sources = cudf.Series(M['0'])
    >>> destinations = cudf.Series(M['1'])
    >>> G = cugraph.Graph()
    >>> G.add_edge_list(sources, destinations, None)
    >>> k_max = cugraph.ktruss_max(G)
    """

    k_max = ktruss_max_wrapper.ktruss_max(G)

    return k_max
