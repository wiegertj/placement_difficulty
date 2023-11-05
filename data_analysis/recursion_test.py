from ete3 import Tree
import sys
sys.setrecursionlimit(2000)

tree_filepath = "/hits/fast/cme/wiegerjs/corona_test_new_config.raxml.bestTree"
with open(tree_filepath, "r") as tree_file:
    tree_str = tree_file.read()
    phylo_tree_reference = Tree(tree_str)
    branch_id_counter = 0
    for node in phylo_tree_reference.traverse():
        branch_id_counter += 1
        if not node.is_leaf():
            node.__setattr__("name", branch_id_counter)

    current_tree = phylo_tree_reference.copy()