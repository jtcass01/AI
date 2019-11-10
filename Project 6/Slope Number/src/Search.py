class BreadthFirstSearchTree(object):
    def __init__(self):
        self.nodes = {}

    def add_node(self, new_node):
        print("\nAttempting to Add node:")
        print(new_node)

        for layer_key in self.nodes.keys():
            for node_index, node in enumerate(self.nodes[layer_key]):
                if node.vertex.vertex_id == new_node.vertex.vertex_id:
                    print("new_node found within dictionary of nodes.")
                    print("previous node: ")
                    print(node)
                    # vertex has exists within searchtree already, see if new_node
                    # has a shorter route.  If so replace.  If not, ignore node.
                    if new_node.minimum_route < node.minimum_route:
                        print("new_node < node. Adding")
                        self.nodes[layer_key][node_index] = deepcopy(new_node)
                        return True
                    else:
                        print("new_node > node.  Ignoring")
                        return False

        print("new_node not found within dictionary of nodes.")
        # If node does not exist, it is minimum route node by default.
        # If new_node's layer has been created
        if new_node.layer in self.nodes.keys():
            print("new_node's layer is not new.  Appending.")
            # Append the new_node to the dictionary's list at layer L
            self.nodes[str(new_node.layer)].append(new_node)
        else:
            print("new_node's layer is new.  Starting new list.")
            # else, create a new list for layer L's nodes
            self.nodes[str(new_node.layer)] = [new_node]

        return True

    def node_in_tree(self, node):
        for layer_key in self.nodes.keys():
            if node in self.nodes[layer_key]:
                return True
        return False

    def vertex_in_tree(self, vertex):
        for layer_key in self.nodes.keys():
            for node in self.nodes[layer_key]:
                if node.vertex.vertex_id == vertex.vertex_id:
                    return True
        return False

    def display(self):
        for layer_key in self.nodes.keys():
            print()
            print("===  Layer: ", layer_key, "===")
            for node in self.nodes[layer_key]:
                print(node)
                print()

    def plot(self):
        x = list([])
        y = list([])
        plots = list([])

        # Iterate over vertices, retrieving x and y coordinates
        for layer_key in self.nodes.keys():
            for node in self.nodes[layer_key]:
                x.append(node.vertex.x)
                y.append(node.vertex.y)

        # Show the graph with a legend
        plt.legend(loc=2, fontsize='small')
        plt.show()

    class Node(object):
        def __init__(self, node_id, vertex, layer, minimum_route):
            self.node_id = node_id
            self.vertex = vertex
            self.layer = layer
            self.adjacent_nodes = list([])
            self.minimum_route = deepcopy(minimum_route)

        def __str__(self):
            string = "node_id: " + str(self.node_id) + "\nvertex: " + str(self.vertex) + "\nlayer: " + self.layer + "\nadjacent_nodes [vertex_ids]: "

            string += "["

            for node in self.adjacent_nodes:
                string += str(node.vertex.vertex_id) + ","

            string += "]"

            string += "\nminimum_route:" + str(self.minimum_route)

            return string


        def __eq__(self, other):
            return self.vertex.vertex_id == other.vertex.vertex_id

class DepthFirstSearchStack(object):
    def __init__(self):
        self.node_stack = list([])
        self.finished_nodes = list([])
        self.finished_vertices = list([])

    def __str__(self):
        string = "==== Depth First Search Stack ===="
        string += "\nNode Stack: \n"
        for node in self.node_stack:
            string += str(node) + "\n"
        string += "\nFinished Nodes: \n"
        for node in self.finished_nodes:
            string += str(node) + "\n"
        string += "\nFinished Vertices: \n"
        for vertex in self.finished_vertices:
            string += str(vertex) + "\n"
        return string

    def get_unfinished_adjacent_vertices(self, adjacent_vertices):
        return [vertex for vertex in adjacent_vertices if vertex not in self.finished_vertices]

    def push(self, vertex, current_route):
        matching_vertices = [node.vertex for node in self.node_stack if node.vertex == vertex]

        if vertex not in matching_vertices:
            node = DepthFirstSearchStack.Node(vertex.vertex_id, vertex, current_route)
            self.node_stack.append(node)

    def get_path_to_finished_vertex_id(self, vertex_id):
        return [node.minimum_route for node in self.finished_nodes if node.vertex.vertex_id == vertex_id][0]

    def pop(self):
        top = self.node_stack[-1]
        self.node_stack.remove(top)
        return top

    def node_complete(self, node):
        print("Node marked as complete:" + str(node))

        self.finished_nodes.append(node)
        self.finished_vertices.append(node.vertex)
        node.finished = True

    class Node(object):
        def __init__(self, node_id, vertex, minimum_route):
            self.node_id = node_id
            self.vertex = vertex
            self.minimum_route = deepcopy(minimum_route)
            self.finished = False

        def __str__(self):
            return "node_id: " + str(self.node_id) + " finished: " + str(self.finished) + "\nvertex: " + str(self.vertex) + "\nminimum_route: " + str(self.minimum_route)
