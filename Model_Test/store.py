# storing the embeddings for either deallocation or 
# all indices are integer
class embeddings:
    def __init__(self, num):
        self.vnr_id = num
        self.vnode_weights = dict()
        self.link_weights = dict()
        self.link_path = dict()
        self.snode_map = dict()

    def store_link(self, vnode1, vnode2, weight, path):
        self.link_path[(vnode1,vnode2)] = path
        self.link_path[(vnode2,vnode1)] = path
        self.link_weights[(vnode1,vnode2)] = weight
        self.link_weights[(vnode2,vnode1)] = weight
        # print(f"Link between {vnode1},{vnode2} on path {path} with weight {self.link_weights[(vnode1,vnode2)]}")

    def store_snode(self, vnode, snode, weight):
        self.snode_map[vnode] = snode
        self.vnode_weights[vnode] = weight

    def clear(self):
        self.vnode_weights = dict()
        self.link_weights = dict()
        self.link_path = dict()
        self.snode_map = dict()