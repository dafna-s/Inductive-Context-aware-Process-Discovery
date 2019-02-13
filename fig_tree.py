import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import igraph
from igraph import *

import config


class Tree:
    def __init__(self, node, data, row, silhouette, log_rows, children=None, parent=None):
        self.name = node # node number
        self.children = []
        self.data = data
        self.row = row
        self.sil = silhouette
        self.log_rows = log_rows
        self.parent = -1

    def add_child(self, node):
        self.children.append(node)
        node.parent = self


def create_tree(tree, sil=None):
    counter_nodes = 0
    row = 0
    sil_i = 0
    while len(tree) > 0:
        #     print('tree:',tree)
        next_operator = [tree.find('('), tree.find(')'), tree.find(',')]
        operator = min(operator for operator in next_operator if operator > -1)
        #     print (config.process_tree[operator])
        operator_name = tree[:operator]
        if operator_name == '':
            row -= 1 if tree[:1] == ')' else 0
            tree = tree[operator + 1:]

            continue
        if operator_name.lower() in config.operators and not sil == None:
            silhouette = sil[sil_i][0]
            log_rows = sil[sil_i][1]
            sil_i += 1
        else:
            silhouette = None
            log_rows = None
        config.process_tree.append(Tree(counter_nodes, operator_name, row, silhouette, log_rows))

        #     print(config.process_tree[-1].name, config.process_tree[-1].data, config.process_tree[-1].row)

        if len(config.process_tree) > 1:
            for i in range(len(config.process_tree) - 2, -1, -1):
                if config.process_tree[i].row + 1 == config.process_tree[-1].row:
                    break
            config.process_tree[i].add_child(config.process_tree[-1])

        counter_nodes += 1
        if tree[operator] == '(':
            change_row = 1
        elif tree[operator] == ')':
            change_row = -1
        elif tree[operator] == ',':
            change_row = 0
        row += change_row
        tree = tree[operator + 1:]


def create_tree_fig():
    nr_vertices = len(config.process_tree)
    v_label = list(map(str, range(nr_vertices)))
    activity_nodes = []
    for i in range(nr_vertices):
        if config.process_tree[i].data.lower() == 'xor':
            v_label[i] = u'\u00D7'
        elif config.process_tree[i].data.lower() == 'and':
            v_label[i] = u'\u2227'
        elif config.process_tree[i].data.lower() == 'seq':
            v_label[i] = u'\u2192'
        elif config.process_tree[i].data.lower() == 'loop':
            v_label[i] = u'\u21BA'
        else:
            v_label[i] = config.process_tree[i].data
            activity_nodes.append(i)

    def set_position(i, horizontal):
        if not config.process_tree[i].children:
            return
        parent_row = config.process_tree[i].row
        range_child = [horizontal - pow(1 / (parent_row + 1), 2), horizontal + pow(1 / (parent_row + 1), 2)]
        # print('range_child ', range_child)
        children_num = len(config.process_tree[i].children)
        # print('children_num ', children_num)
        step = (range_child[1] - range_child[0]) / (children_num - 1)
        # print((range_child[1] - range_child[0]) / (children_num - 1))
        for k in range(children_num):
            position[config.process_tree[i].children[k].name] = [range_child[0] + (k * step),
                                                                 config.process_tree[i].children[k].row]
            E.append((config.process_tree[i].name, config.process_tree[i].children[k].name))
            set_position(config.process_tree[i].children[k].name, range_child[0] + (k * step))

    E = []
    i = 0
    position = {config.process_tree[i].name: [i, config.process_tree[i].row]}
    set_position(i, 0.0)

    G = Graph.Tree(nr_vertices, 2)  # 2 stands for children number
    lay = G.layout('rt')

    Y = [lay[k][1] for k in range(nr_vertices)]
    M = max(Y)

    Xn_a = []
    Yn_a = []
    Xn_o = []
    Yn_o = []
    L = len(position)
    for k in range(L):
        if k in activity_nodes:
            Xn_a.append(position[k][0])
            Yn_a.append(2 * M - position[k][1])
        else:
            Xn_o.append(position[k][0])
            Yn_o.append(2 * M - position[k][1])

    Xe = []
    Ye = []
    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye += [2 * M - position[edge[0]][1], 2 * M - position[edge[1]][1], None]

    labels = v_label

    lines = go.Scatter(x=Xe,
                       y=Ye,
                       mode='lines',
                       line=dict(color='rgb(210,210,210)', width=1),
                       hoverinfo='none'
                       )
    operators_dots = go.Scatter(x=Xn_o,
                                y=Yn_o,
                                mode='markers',
                                name='',
                                marker=dict(symbol='dot',
                                            size=30,
                                            color='rgb(255,255,255)',  # '#DB4551',
                                            line=dict(color='rgb(255,255,255)', width=1)
                                            ),
                                text=labels,
                                hoverinfo='text',
                                opacity=0.1
                                )
    activity_dots = go.Scatter(x=Xn_a,
                               y=Yn_a,
                               mode='markers',
                               name='',
                               marker=dict(symbol='dot',
                                           size=30,
                                           color='rgb(255,255,255)',  # '#DB4551',
                                           line=dict(color='rgb(255,255,255)', width=1)
                                           ),
                               text=labels,
                               textfont=dict(
                                   color='#DB4551'),
                               hoverinfo='text',
                               opacity=0.01
                               )

    def make_annotations(pos, text, font_size=10, font_color='rgb(0,0,0)'):
        L = len(list(pos))
        if len(list(text)) != L:
            raise ValueError('The lists pos and text must have the same len')
        annotations = go.Annotations()
        for k in range(L):
            annotations.append(
                go.Annotation(
                    text=labels[k],  # or replace labels with a different list for the text within the circle
                    x=pos[k][0], y=2 * M - position[k][1],
                    xref='x1', yref='y1',
                    font=dict(color=font_color, size=font_size),
                    showarrow=False)
            )
        return annotations

    axis = dict(showline=False,  # hide axis line, grid, ticklabels and  title
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                )
    title = config.data_file, ' silhouette=', config.silhouette_threshold
    layout = dict(title=title,
                  annotations=make_annotations(position, v_label),
                  font=dict(size=12),
                  showlegend=False,
                  xaxis=go.XAxis(axis),
                  yaxis=go.YAxis(axis),
                  margin=dict(l=40, r=40, b=85, t=100),
                  hovermode='closest',
                  plot_bgcolor='rgb(250,250,250)',
                  paper_bgcolor='rgb(250,250,250)'
                  )
    data = go.Data([lines, activity_dots, operators_dots])
    fig = dict(data=data, layout=layout)
    fig['layout'].update(annotations=make_annotations(position, v_label))
    py.iplot(fig, filename='Tree-Reingold-Tilf')
    plotly.offline.plot(fig, filename='Tree.html')

    print('fig name: Tree.html created')
