from datetime import datetime
from pathlib import Path
from typing import Sequence, Set, Union

from py2neo import Graph, Node, NodeMatcher, RelationshipMatcher, ConnectionUnavailable


# from config.config import Config
# from utils.logger import Logger


# 编写一个GraphDao类，该类包含与图相关的数据库操作方法。
class GraphDao(object):
    """
    连接图数据库neo4j
    """

    # _logger: Logger = Logger(Path(__file__).name)

    def __init__(self, args):
        self.args = args
        # 读取yaml配置
        # self.__url = Config.get_instance().get_with_nested_params("database", "neo4j", "url")
        # self.__username = Config.get_instance().get_with_nested_params("database", "neo4j", "username")
        # self.__password = Config.get_instance().get_with_nested_params("database", "neo4j", "password")
        self.__url = 'http://172.31.169.49:7474'
        self.__username = 'neo4j'
        self.__password = 'yf939924'

        self.__connect_graph()
        self.__meta_node_id = 'meta-001'
        # self.__meta_node_id = '1'

        # 创建节点匹配器
        self.__node_matcher = NodeMatcher(self.__graph) if self.__graph else None

        # 创建关系匹配器
        self.__relationship_matcher = RelationshipMatcher(self.__graph) if self.__graph else None

    # @staticmethod
    # def ensure_connection(function):
    #     def wrapper(*args, **kwargs):
    #         if not args[0].__graph:
    #             GraphDao._logger.warning("Graph database connection is not available.")
    #             return None
    #         return function(*args, **kwargs)
    #
    #     return wrapper

    def __connect_graph(self):
        try:
            self.__graph = Graph(self.__url, auth=(self.__username, self.__password))
        except ConnectionUnavailable as e:
            # self._logger.warning(f"Failed to connect to Neo4j graph database: {e}")
            self.__graph = None

    # @ensure_connection
    def create_node(self, label, properties):
        """
        创建节点
        :param label: 节点标签
        :param properties: 节点属性
        :return: 创建的节点对象
        """
        node = Node(label, **properties)
        self.__graph.create(node)
        return node

    # @ensure_connection
    def create_meta_node(self, version: int):
        """
        创建元节点
        :return: 创建的元节点对象,记录创建时间等信息
        """
        meta_node = Node(
            'Meta',
            id=self.__meta_node_id,
            title='grag meta node',
            text='store some meta info',
            # python获取当前时间
            timestamp=datetime.now(),
            version=version,
            status='active',
        )
        self.__graph.create(meta_node)
        return meta_node

    # @ensure_connection
    def query_meta_node(self):
        return self.__node_matcher.match('Meta', id=self.__meta_node_id).first()

    def query_data_version(self):
        return int(self.query_meta_node()['version'])

    def update_version(self):
        """
        更新元节点的版本号
        return: 更新后的元节点对象
        """
        meta_node = self.query_meta_node()
        if meta_node:
            meta_node['version'] += 1
            self.__graph.push(meta_node)
            return meta_node
        else:
            raise Exception('Meta node not found')

    def delete_node(self, node):
        ...

    def delete_relationship(self, relationship):
        ...

    def update_node(self, node, properties):
        ...

    def update_relationship(self, relationship, properties):
        ...

    # @ensure_connection
    def query_node(self, *label, **properties):
        return self.__node_matcher.match(*label, **properties)

    def query_node_by_id(self, input_id: str):
        # 查询内部 ID 为 1 的节点
        query = f"""
                MATCH (n {{id: {input_id}}})
                RETURN n
                """

        # 执行查询并获取结果
        results = self.__graph.run(query).data()
        return results

    # @ensure_connection
    def query_relationship(self, nodes: Union[Sequence, Set, None] = None, r_type=None, **properties):
        return self.__relationship_matcher.match(nodes, r_type, **properties)

    def query_triples_by_node(self, node: Node, need_node_in_relation: list):
        triples_names = []
        triples_ids = []
        # 如果找到了节点，获取所有一跳 OUTGOING 三元组
        if not self.args.use_reverse_retrieval:
            # 正向检索
            query = "MATCH (s)-[r]->(t) WHERE id(s) = $node_id RETURN s,r,t"
            results = self.__graph.run(query, node_id=node.identity)

            for rel in results.data():
                # 删除关系中，不存在必须节点的情况
                if len(need_node_in_relation) > 0:
                    iscontinue = False
                    for neednode in need_node_in_relation:
                        # 两边加空格，保证是单独的一个节点名称，而不是长的名称包含短的名称的现象
                        if f' {neednode} ' not in rel['r']['condition']:
                            iscontinue = True
                            break
                    if iscontinue:
                        continue

                star_node_name = rel['s']['name']
                star_node_id = rel['s']['id']
                # 这种方式，会在字符串开头添加一个空格，去除掉空格
                rel_name = rel['r']['condition']
                rel_history = rel['r']['history']
                # rel_name = rel.__class__.__name__
                rel_id = rel['r']['rel_id']

                end_node_name = rel['t']['name']
                end_node_id = rel['t']['id']

                # 获取关系的ID和属性
                triples_names.append(f'{star_node_name}|{rel_name}|{rel_history}|{end_node_name}')
                triples_ids.append(f'{star_node_id}|{rel_id}|{end_node_id}')
                # 删除超过20个子节点的路径,这里是因为前序路走错了.子节点太多,后续会过于耗时,建训练数据阶段,只保留了20个子节点的路径
                if len(triples_names) > 21:
                    return [], []
        else:
            # 逆向检索
            query = "MATCH (t)<-[r]-(s) WHERE id(t) = $node_id RETURN s,r,t"
            results = self.__graph.run(query, node_id=node.identity)
            for rel in results.data():
                # 删除关系中，不存在必须节点的情况
                if len(need_node_in_relation) > 0:
                    iscontinue = False
                    for neednode in need_node_in_relation:
                        # 两边加空格，保证是单独的一个节点名称，而不是长的名称包含短的名称的现象
                        if f' {neednode} ' not in rel['r']['condition'] and f' {neednode},' not in rel['r']['condition']:
                            iscontinue = True
                            break
                    if iscontinue:
                        continue

                star_node_name = rel['s']['name']
                star_node_id = rel['s']['id']
                # 这种方式，会在字符串开头添加一个空格，去除掉空格
                rel_name = rel['r']['condition']
                rel_history = rel['r']['history']
                # rel_name = rel.__class__.__name__
                rel_id = rel['r']['rel_id']

                end_node_name = rel['t']['name']
                end_node_id = rel['t']['id']

                # 获取关系的ID和属性
                triples_names.append(f'{star_node_name}|{rel_name}|{rel_history}|{end_node_name}')
                triples_ids.append(f'{star_node_id}|{rel_id}|{end_node_id}')
                # 删除超过20个子节点的路径,这里是因为前序路走错了.子节点太多,后续会过于耗时,建训练数据阶段,只保留了20个子节点的路径
                if len(triples_names) > 21:
                    return [], []

        return triples_names, triples_ids

    # @ensure_connection
    def query_relationship_by_2person_name(self, first_person, second_person):
        rel = self.__graph.run(
            f"match(:`entity`{{name:'{first_person}'}})-[r]-(:`entity`{{name:'{second_person}'}}) return r, type(r)").data()
        return rel

    def query_path(self, start_node, end_node, rel_type, rel_properties):
        ...

    def query_all(self):
        ...

    # @ensure_connection
    def run_cypher(self, query):
        result = self.__graph.run(query)
        return result
