from __future__ import annotations

import enum
import json
from abc import ABC, abstractmethod
from typing import Dict

from fuzzylogicpy.core.elements import Node
from fuzzylogicpy.parser.expression_parser import ExpressionParser


def query_to_json(query: Query) -> str:
    return json.dumps(query, default=lambda o: o.__dict__, sort_keys=False, indent=4)


def query_from_json(query_string: str) -> Query:
    dict_ = json.loads(query_string)
    if isinstance(dict_, Dict) and 'type' in dict_.keys():
        print(dict_['type'])
        if dict_['type'] == str(QueryType.EVALUATION):
            dict_.pop('type')
            return EvaluationQuery(**dict_)
        elif dict_['type'] == str(QueryType.DISCOVERY):
            dict_.pop('type')
            return DiscoveryQuery(**dict_)
        else:
            raise RuntimeError('Invalid object: ' + str(dict_))
    else:
        raise RuntimeError('Invalid object: ' + str(dict_))


class QueryType(str, enum.Enum):
    EVALUATION = 'EVALUATION'
    DISCOVERY = 'DISCOVERY'

    def __repr__(self):
        if self == QueryType.EVALUATION:
            return 'EVALUATION'
        elif self == QueryType.DISCOVERY:
            return 'DISCOVERY'
        else:
            raise RuntimeError("Invalid")

    def __str__(self):
        return self.__repr__()


class LogicType(str, enum.Enum):
    GMBC = 'GMBC'
    ZADEH = 'ZADEH'

    def __repr__(self):
        if self == LogicType.GMBC:
            return 'GMBC'
        elif self == LogicType.ZADEH:
            return 'ZADEH'
        else:
            raise RuntimeError("Invalid")

    def __str__(self):
        return self.__repr__()


class Query(ABC):
    def __init__(self, db_uri: str, out_file: str, states: Dict, logic: LogicType, expression: str):
        self.type = None
        self.db_uri = db_uri
        self.out_file = out_file
        self.states = states
        self.logic = logic
        self.expression = expression

    @abstractmethod
    def get_tree(self) -> Node:
        pass

    def name(self) -> str:
        return str(self.__class__.__name__)


class EvaluationQuery(Query):
    def __init__(self, db_uri: str, out_file: str, states: Dict, logic: LogicType, expression: str):
        super().__init__(db_uri, out_file, states, logic, expression)
        self.type = QueryType.EVALUATION

    def get_tree(self) -> Node:
        return ExpressionParser(self.expression, self.states, {}).parser()


class DiscoveryQuery(Query):
    def get_tree(self) -> Node:
        return ExpressionParser(self.expression, self.states, self.generators).parser()

    def __init__(self, db_uri: str, out_file: str, states: dict, logic: LogicType, expression: str, generators: Dict,
                 num_pop: int, num_iter: int, num_result: int, mut_percentage: float, adj_num_pop: int = None,
                 adj_num_iter: int = None, adj_min_truth_value: float = None):
        super().__init__(db_uri, out_file, states, logic, expression)
        self.type = QueryType.DISCOVERY
        self.generators = generators
        self.num_pop = num_pop
        self.num_iter = num_iter
        self.num_result = num_result
        self.mut_percentage = mut_percentage
        self.adj_num_iter = adj_num_iter
        self.adj_num_pop = adj_num_pop
        self.adj_min_truth_value = adj_min_truth_value
